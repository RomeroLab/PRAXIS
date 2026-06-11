import os
import sys
import yaml
import json
import time
import logging
import hashlib
import subprocess
import pandas as pd
from enum import Enum
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from data_transfer import SFTPTransfer
from notify import send_sms




# Status Tracking
class LabState(Enum):
    IDLE = "Waiting for new sequence"
    SEQUENCE_RECEIVED = "New sequence received, generating instructions"
    MONITORING_EVAGREEN = "Monitoring evagreen data"
    EXPERIMENTING = "Robot performing DNA assembly, amplification, expression, assay"
    PROCESSING = "Processing plate data"
    ERROR = "Error encountered"

class PlateDataHandler(FileSystemEventHandler):
    def __init__(self, data_dir: str, input_filename: str, output_filename: str, controller=None):
        self.data_dir = data_dir
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.logger = logging.getLogger('lab_automation')
        self.last_processed_hash = None
        self.transfer = SFTPTransfer()
        self.controller = controller

        # Log initialization
        self.logger.info(f"Initialized PlateDataHandler:")
        self.logger.info(f"  data_dir: {data_dir}")
        self.logger.info(f"  input_file: {input_filename}")
        self.logger.info(f"  output_file: {output_filename}")

    def get_file_hash(self, filepath):
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def process_and_transfer(self):
        """Process plate data, transfer, archive raw data and layout, delete query."""
        # Define the input directory for raw_plate_data.csv for processing
        input_dir_for_processing = os.path.join(self.data_dir, 'raw_plate_data')
        # Define the full path to raw_plate_data.csv within raw_plate_data directory for archiving
        raw_plate_data_source_path = os.path.join(input_dir_for_processing, self.input_filename)

        output_path = os.path.join(self.data_dir, self.output_filename) # phenotype.json path
        sequence_file_path = os.path.join(self.data_dir, 'sequence_query.txt')
        plate_layout_src_path = 'plate_layout.json' # Assuming it's in the root relative to where script is run

        try:
            # Process the data (phenotype generation)
            python_path = sys.executable
            subprocess.run([
                python_path, 'process_plate_data.py',
                '--input', input_dir_for_processing, # Pass the directory to process_plate_data.py
                '--output', output_path,
                '--sequence_file', sequence_file_path
            ], check=True)

            time.sleep(1)
            
            # Update sequence tracking CSV
            self.update_sequence_tracking_csv(output_path)
            
            # Transfer to GPU server
            if self.transfer.connect():
                self.transfer.transfer_file(output_path)
                self.transfer.close()
                self.logger.info("Processed data and transferred to GPU server")
                # Send phenotype summary as SMS with chimera IDs from plate layout
                try:
                    with open(output_path, 'r') as f:
                        phenotype = json.load(f)
                    # Get chimera IDs from plate_layout.json (row A-F, col 1)
                    chimera_ids = {}
                    if os.path.exists('plate_layout.json'):
                        with open('plate_layout.json', 'r') as f:
                            layout = json.load(f)
                        for i, row in enumerate('ABCDEF'):
                            cid = layout.get(f"{row}1", "").split("-")[0]
                            if cid:
                                chimera_ids[i] = cid
                    lines = []
                    for i, (seq, data) in enumerate(phenotype.items()):
                        if not data.get('valid', False):
                            continue
                        m = data.get('measurements', [0, 0, 0])
                        cid = chimera_ids.get(i, seq[:8])
                        lines.append(f"{cid}: Aglu:{m[0]:.2f} Axyl:{m[1]:.2f} Aman:{m[2]:.2f}")
                    send_sms("\n".join(lines))
                except Exception:
                    send_sms("AutoLab: Plate data processed, transferred to GPU")

                # --- Archive raw data and plate layout into timestamped folder --- 
                try:
                    # Construct archive directory name with date and enzyme IDs from plate_layout.json
                    date_str = datetime.now().strftime("%Y%m%d")
                    enzyme_ids_str = "XXXXXXXX_XXXXXXXX_XXXXXXXX_XXXXXXXX_XXXXXXXX_XXXXXXXX" # Default if layout not found
                    
                    plate_layout_path = 'plate_layout.json' # Assuming it's in the root
                    if os.path.exists(plate_layout_path):
                        with open(plate_layout_path, 'r') as f_layout:
                            layout_data = json.load(f_layout)
                        # Extract enzyme IDs (one per row A-F) from the first column of each row
                        row_wells = ["A1", "B1", "C1", "D1", "E1", "F1"]
                        enzyme_ids = [layout_data.get(w, "XXXXXXXX-glu-R1").split("-")[0] for w in row_wells]
                        enzyme_ids_str = "_".join(enzyme_ids)
                    else:
                        self.logger.warning(f"plate_layout.json not found at {plate_layout_path}, using default enzyme IDs for archive name.")

                    archive_folder_name = f"{date_str}_{enzyme_ids_str}"
                    archive_parent_dir = os.path.join(self.data_dir, 'archive')
                    timestamp_archive_dir = os.path.join(archive_parent_dir, archive_folder_name)
                    os.makedirs(timestamp_archive_dir, exist_ok=True)
                    
                    # Define destination paths
                    raw_data_archive_path = os.path.join(timestamp_archive_dir, self.input_filename)
                    layout_archive_path = os.path.join(timestamp_archive_dir, os.path.basename(plate_layout_src_path))
                    phenotype_archive_path = os.path.join(timestamp_archive_dir, self.output_filename)
                    evagreen_filename = 'raw_evagreen_data.csv'
                    evagreen_source_path = os.path.join(self.data_dir, 'raw_evagreen_data', evagreen_filename)
                    evagreen_archive_path = os.path.join(timestamp_archive_dir, evagreen_filename)
                    
                    # Move raw plate data from the raw_plate_data subdirectory
                    if os.path.exists(raw_plate_data_source_path):
                        os.rename(raw_plate_data_source_path, raw_data_archive_path)
                        self.logger.info(f"Archived raw plate data from {raw_plate_data_source_path} to: {raw_data_archive_path}")
                    else:
                         self.logger.warning(f"Raw plate data file not found for archiving at: {raw_plate_data_source_path}")

                    # Move plate layout
                    if os.path.exists(plate_layout_src_path):
                        os.rename(plate_layout_src_path, layout_archive_path)
                        self.logger.info(f"Archived plate layout to: {layout_archive_path}")
                    else:
                        self.logger.warning(f"Plate layout file not found for archiving: {plate_layout_src_path}")
                    
                    # Move phenotype file
                    if os.path.exists(output_path):
                        os.rename(output_path, phenotype_archive_path)
                        self.logger.info(f"Archived phenotype data to: {phenotype_archive_path}")
                    else:
                        self.logger.warning(f"Phenotype file not found for archiving: {output_path}")

                    # Move evagreen data file
                    if os.path.exists(evagreen_source_path):
                        os.rename(evagreen_source_path, evagreen_archive_path)
                        self.logger.info(f"Archived evagreen data from {evagreen_source_path} to: {evagreen_archive_path}")
                    else:
                        self.logger.warning(f"Evagreen data file not found for archiving at: {evagreen_source_path}")

                except OSError as e:
                    self.logger.error(f"Error creating/moving files to timestamped archive directory: {e}")
                # --- End Archiving --- 

                # Delete sequence query file (original behavior)
                if os.path.exists(sequence_file_path):
                    try:
                        os.remove(sequence_file_path)
                        self.logger.info(f"Deleted sequence query file: {sequence_file_path}")
                    except OSError as e:
                         self.logger.error(f"Error deleting sequence query file: {e}")

                # Stop plate monitoring and restart sequence monitoring
                if self.controller:
                    if self.controller.plate_observer:
                        self.controller.plate_observer.stop()
                        self.controller.plate_observer = None
                    self.controller.start_sequence_monitoring()
                    send_sms("AutoLab: Run complete, ready for next sequences")
                return True
            else:
                self.logger.error("Failed to connect to GPU server")
                send_sms("AutoLab ERROR: GPU server connection failed")
                return False


        except Exception as e:
            self.logger.error(f"Error processing/transferring data: {e}")
            send_sms("AutoLab ERROR: plate data processing failed")

    def update_sequence_tracking_csv(self, phenotype_file_path):
        """Update running CSV of all sequences tested with their measurements."""
        try:
            csv_file_path = os.path.join(self.data_dir, 'sequence_tracking.csv')
            
            # Read the phenotype data from JSON
            with open(phenotype_file_path, 'r') as f:
                phenotype_data = json.load(f)
            
            # Create a dataframe with existing data if CSV exists
            if os.path.exists(csv_file_path):
                existing_df = pd.read_csv(csv_file_path)
            else:
                existing_df = pd.DataFrame(columns=['sequence', 'A1', 'A2', 'A3', 'valid', 'agent'])
            
            # Process each sequence in the new phenotype data
            new_entries = []
            for agent_idx, (sequence, data) in enumerate(phenotype_data.items(), 1):
                # Skip if this sequence is already in our tracking CSV
                if sequence in existing_df['sequence'].values:
                    continue
                    
                # Extract measurements and valid flag
                measurements = data.get('measurements', [0, 0, 0])
                valid = data.get('valid', False)
                
                # Create new entry
                new_entries.append({
                    'sequence': sequence,
                    'A1': measurements[0] if len(measurements) > 0 else 0,
                    'A2': measurements[1] if len(measurements) > 1 else 0,
                    'A3': measurements[2] if len(measurements) > 2 else 0,
                    'valid': valid,
                    'agent': agent_idx  # 1, 2, or 3 based on order in JSON
                })
            
            # Add new entries to existing dataframe
            if new_entries:
                new_df = pd.DataFrame(new_entries)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_csv(csv_file_path, index=False)
                self.logger.info(f"Added {len(new_entries)} new sequences to tracking CSV")
            
        except Exception as e:
            self.logger.error(f"Error updating sequence tracking CSV: {e}")

    def on_modified(self, event):
        if event.src_path.endswith(self.input_filename):
            current_hash = self.get_file_hash(event.src_path)
            if current_hash == self.last_processed_hash:
                return
            
            self.logger.info("Plate data modified, processing...")
            self.process_and_transfer()
            self.last_processed_hash = current_hash

class EvagreenHandler(FileSystemEventHandler):
    def __init__(self, controller):
        self.controller = controller
        self.logger = logging.getLogger('lab_automation')
        
    def on_modified(self, event):
        if event.src_path.endswith('raw_evagreen_data.csv'):
            self.logger.info("Evagreen data updated, processing assemblies...")
            try:
                # Use the current Python interpreter (from your conda environment)
                python_path = sys.executable
                subprocess.run([python_path, 'update_valid_assemblies.py'], check=True)
                
                # Stop evagreen monitoring and start plate monitoring
                if self.controller.evagreen_observer:
                    self.controller.evagreen_observer.stop()
                    self.controller.evagreen_observer = None
                
                self.controller.start_plate_monitoring()
                self.logger.info("Transitioned to plate monitoring")

                # Count how many sequences passed dsDNA check
                try:
                    with open('data/phenotype.json', 'r') as f:
                        phenotype = json.load(f)
                    passed = sum(1 for v in phenotype.values() if v.get('valid', False))
                    total = len(phenotype)
                    send_sms(f"AutoLab: Evagreen QC done, {passed}/{total} sequences passed dsDNA check")
                except Exception:
                    send_sms("AutoLab: Evagreen QC done, plate monitoring started")

            except Exception as e:
                self.logger.error(f"Error processing assemblies: {e}")
                self.controller.status = LabState.ERROR
                send_sms("AutoLab ERROR: assembly processing failed")

class DNATracker:
    def __init__(self, inventory_dir='inventories'):
        self.inventory_dir = inventory_dir
        self.inventory_file = os.path.join(inventory_dir,'dna_inventory.json')
        os.makedirs(inventory_dir,exist_ok=True)
        self.inventory = self.load_inventory()
        
    def load_inventory(self):
        """Load or initialize DNA fragment inventory"""
        try:
            with open(self.inventory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Initialize with 48 fragments, 150µL each
            inventory = {}
            for p in range(1, 7):  # 6 parents
                for f in range(8):  # 8 fragments each
                    fragment_id = f"p{p}f{f}"
                    # Format well as "A01", "B01", etc.
                    row = chr(66 + (p-1))  # B + offset
                    col = str(f + 3).zfill(2)  # Start at 03
                    well = f"{row}{col}"
                    
                    inventory[fragment_id] = {
                        "well": well,
                        "volume": 150,  # Initial volume in µL
                        "history": []
                    }
            
            self.save_inventory(inventory)
            return inventory
            
    def save_inventory(self, inventory=None):
        """Save current inventory state"""
        if inventory is None:
            inventory = self.inventory
        with open(self.inventory_file, 'w') as f:
            json.dump(inventory, f, indent=2)
            
    def get_well_formats(self, well):
        """Convert between A01 and A1 formats to ensure matching"""
        row = well[0]
        col = well[1:]
        return [well, f"{row}{int(col)}", f"{row}{int(col):02d}"]

    def update_volumes(self, worklist_file):
        """Update volumes based on worklist"""
        df = pd.read_csv(worklist_file)

        # Track all updates for this run
        updates = []
        timestamp = datetime.now().isoformat()
        
        for _, row in df.iterrows():
            well = row['Source_Well']
            volume_used = float(row['Volume'])
            well_formats = self.get_well_formats(well)
            
            # Find fragment id from well
            fragment_id = None
            for fid, data in self.inventory.items():
                if data['well'] in well_formats:
                    fragment_id = fid
                    break
            
            if fragment_id:
                # Update volume
                current_vol = self.inventory[fragment_id]['volume']
                new_vol = current_vol - volume_used
                
                # Record the update
                update = {
                    'timestamp': timestamp,
                    'volume_used': volume_used,
                    'volume_remaining': new_vol
                }
                
                self.inventory[fragment_id]['volume'] = new_vol
                self.inventory[fragment_id]['history'].append(update)
                
                updates.append({
                    'fragment': fragment_id,
                    'well': well,
                    'volume_used': volume_used,
                    'volume_remaining': new_vol
                })
            else:
                print(f"No matching fragment found for well {well}")
        
        # Save updated inventory
        self.save_inventory()
        return updates
        
    def export_to_csv(self):
        """Export current inventory to CSV"""
        csv_path = os.path.join(self.inventory_dir, 'dna_inventory.csv')
        rows = []
        for fid, data in self.inventory.items():
            rows.append({
                'Fragment_ID': fid,
                'Well': data['well'],
                'Volume_Remaining': data['volume'],
                'Last_Updated': data['history'][-1]['timestamp'] if data['history'] else 'Never'
            })
            
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    def refill_dna(self, fragment_id=None, well=None, volume_added=200):
        """
        Refill DNA fragment wells with specified volume
        Can use either fragment_id (e.g., 'p1f0') or well location (e.g., 'A01')
        """
        timestamp = datetime.now().isoformat()
        updates = []

        # Handle single refill
        if fragment_id or well:
            if fragment_id and fragment_id in self.inventory:
                target = fragment_id
            else:
                # Find fragment_id from well
                well_formats = self.get_well_formats(well)
                target = None
                for fid, data in self.inventory.items():
                    if data['well'] in well_formats:
                        target = fid
                        break
            
            if target:
                current_vol = self.inventory[target]['volume']
                new_vol = current_vol + volume_added
                
                # Record the update
                self.inventory[target]['volume'] = new_vol
                self.inventory[target]['history'].append({
                    'timestamp': timestamp,
                    'volume_added': volume_added,
                    'volume_total': new_vol
                })
                
                updates.append({
                    'fragment': target,
                    'well': self.inventory[target]['well'],
                    'volume_added': volume_added,
                    'volume_total': new_vol
                })
                print(f"Refilled {target} with {volume_added}µL. New total: {new_vol}µL")
            else:
                print(f"No matching fragment found for {fragment_id or well}")
        
        self.save_inventory()
        return updates
    
    def print_inventory_report(self):
        """Print current inventory status"""
        print("\nDNA Fragment Inventory Report")
        print("=" * 50)
        print(f"{'Fragment':<10} {'Well':<8} {'Volume (µL)':<12} {'Status':<10}")
        print("-" * 50)
        
        for fid, data in sorted(self.inventory.items()):
            status = "LOW" if data['volume'] < 50 else "OK"
            print(f"{fid:<10} {data['well']:<8} {data['volume']:<12.1f} {status:<10}")
            
        print("=" * 50)

class RgntTracker:
    def __init__(self, inventory_dir='inventories'):
        self.inventory_dir = inventory_dir
        self.inventory_file = os.path.join(inventory_dir, 'reagent_inventory.json')
        os.makedirs(inventory_dir, exist_ok=True)
        self.inventory = self.load_inventory()


    def load_inventory(self):
        """Load or initialize reagent inventory"""
        try:
            with open(self.inventory_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Define the reagent layout as a dictionary mapping well to reagent name
            reagent_map = {
                "C02": "GG enzyme mix",
                "D02": "T4 ligase buffer",
                "E02": "Phusion PCR MM",
                "C03": "Primers", "D03": "Primers", "E03": "Primers",
                "C04": "Primers", "D04": "Primers", "E04": "Primers",
                "C05": "EvaGreen", "D05": "EvaGreen", "E05": "EvaGreen",
                "C06": "EvaGreen", "D06": "EvaGreen", "E06": "EvaGreen",
                "C07": "H2O", "D07": "H2O", "E07": "H2O", "F07": "H2O",
                "C08": "H2O", "D08": "H2O", "E08": "H2O",
                "C09": "E. coli extract", "D09": "E. coli extract",
                "E09": "E. coli extract", "F09": "E. coli extract",
                "C10": "TXTL MM", "D10": "TXTL MM", "E10": "TXTL MM", "F10": "TXTL MM",
                "C11": "TXTL MM", "D11": "TXTL MM", "E11": "TXTL MM",
            }

            inventory = {}
            for well, name in reagent_map.items():
                # Create a unique ID combining name and well for tracking
                reagent_id = f"{name}_{well}" 
                inventory[reagent_id] = {
                    "well": well,
                    "volume": 150, # Initial vol in uL
                    "history": []
                }

            self.save_inventory(inventory)
            return inventory

            
    def save_inventory(self, inventory=None):
        """Save current inventory state"""
        if inventory is None:
            inventory = self.inventory
        with open(self.inventory_file, 'w') as f:
            json.dump(inventory, f, indent=2)
            
    def get_well_formats(self, well):
        """Convert between A01 and A1 formats to ensure matching"""
        row = well[0]
        col = well[1:]
        return [well, f"{row}{int(col)}", f"{row}{int(col):02d}"]

    def update_volumes(self, worklist_file):
        """Update volumes based on worklist, only for Reagents source plate"""
        df = pd.read_csv(worklist_file)


        # Track all updates for this run
        updates = []
        timestamp = datetime.now().isoformat()
        
        # Filter for only Reagents source plate
        reagents_df = df[df['Source_Plate'] == 'Reagents']
        
        for _, row in reagents_df.iterrows():
            well = row['Source_Well']
            volume_used = float(row['Volume'])
            well_formats = self.get_well_formats(well)
            
            # Find reagent id from well
            reagent_id = None
            for rid, data in self.inventory.items():
                if data['well'] in well_formats:
                    reagent_id = rid
                    break
            
            if reagent_id:
                # Update volume
                current_vol = self.inventory[reagent_id]['volume']
                new_vol = current_vol - volume_used
                
                # Record the update
                update = {
                    'timestamp': timestamp,
                    'volume_used': volume_used,
                    'volume_remaining': new_vol
                }
                
                self.inventory[reagent_id]['volume'] = new_vol
                self.inventory[reagent_id]['history'].append(update)
                
                updates.append({
                    'reagent': reagent_id,
                    'well': well,
                    'volume_used': volume_used,
                    'volume_remaining': new_vol
                })
            else:
                print(f"No matching reagent found for well {well}")
        
        # Save updated inventory
        self.save_inventory()
        return updates
        
    def export_to_csv(self):
        """Export current inventory to CSV"""
        csv_path = os.path.join(self.inventory_dir, 'reagent_inventory.csv')
        rows = []
        for rid, data in self.inventory.items():
            rows.append({
                'Reagent_ID': rid,
                'Well': data['well'],
                'Volume_Remaining': data['volume'],
                'Last_Updated': data['history'][-1]['timestamp'] if data['history'] else 'Never'
            })
            
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    def refill_reagent(self, reagent_id=None, well=None, volume_added=200):
        """
        Refill DNA fragment wells with specified volume
        Can use either reagent_id (e.g., 'GGMM') or well location (e.g., 'D05')
        """
        timestamp = datetime.now().isoformat()
        updates = []

        # Handle single refill
        if reagent_id or well:
            if reagent_id and reagent_id in self.inventory:
                target = reagent_id
            else:
                # Find reagent_id from well
                well_formats = self.get_well_formats(well)
                target = None
                for rid, data in self.inventory.items():
                    if data['well'] in well_formats:
                        target = rid
                        break
            
            if target:
                current_vol = self.inventory[target]['volume']
                new_vol = current_vol + volume_added
                
                # Record the update
                self.inventory[target]['volume'] = new_vol
                self.inventory[target]['history'].append({
                    'timestamp': timestamp,
                    'volume_added': volume_added,
                    'volume_total': new_vol
                })
                
                updates.append({
                    'reagent': target,
                    'well': self.inventory[target]['well'],
                    'volume_added': volume_added,
                    'volume_total': new_vol
                })
                print(f"Refilled {target} with {volume_added}µL. New total: {new_vol}µL")
            else:
                print(f"No matching reagent found for {reagent_id or well}")
        
        self.save_inventory()
        return updates
    
    def print_inventory_report(self):
        """Print current inventory status"""
        print("\nReagent Inventory Report")
        print("=" * 50)
        print(f"{'Reagent':<10} {'Well':<8} {'Volume (µL)':<12} {'Status':<10}")
        print("-" * 50)
        
        for rid, data in sorted(self.inventory.items()):
            status = "LOW" if data['volume'] < 50 else "OK"
            print(f"{rid:<10} {data['well']:<8} {data['volume']:<12.1f} {status:<10}")
            
        print("=" * 50)

class LabController:
    def __init__(self, config_path):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler('lab_controller.log')]
        )
        self.logger = logging.getLogger('lab_controller')
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.status = LabState.IDLE
        self.sequence_observer = None
        self.plate_observer = None
        self.evagreen_observer = None  # Uncomment evagreen observer

    class SequenceHandler(FileSystemEventHandler):
        def __init__(self, controller):
            self.controller = controller
            self.sequence_file = os.path.join(
            self.controller.config['paths']['data_dir'],
            self.controller.config['files']['sequence_query']
        )
            
        def on_modified(self, event):
            if event.src_path.endswith(self.sequence_file):
                self.controller.logger.info("New sequence detected")
                self.controller.status = LabState.SEQUENCE_RECEIVED
                if self.controller.generate_lab_files():
                    # Create a preliminary phenotype.json before evagreen monitoring
                    try:
                        phenotype_file = os.path.join(self.controller.config['paths']['data_dir'], self.controller.config['files']['processed_data'])
                        with open(self.sequence_file, 'r') as f:
                            sequences = [line.strip() for line in f if line.strip()]
                        
                        phenotype_data = {}
                        for seq in sequences:
                            phenotype_data[seq] = {"measurements": [0.0, 0.0, 0.0], "valid": True}
                        
                        with open(phenotype_file, 'w') as f:
                            json.dump(phenotype_data, f, indent=4)
                        self.controller.logger.info(f"Created preliminary phenotype file at {phenotype_file}")
                    except Exception as e:
                        self.controller.logger.error(f"Failed to create preliminary phenotype file: {e}")
                        self.controller.status = LabState.ERROR
                        return

                    self.controller.status = LabState.MONITORING_EVAGREEN
                    self.controller.start_evagreen_monitoring()
                    self.controller.logger.info("Started evagreen monitoring")
                    send_sms("AutoLab: Sequences received, worklists generated")

    def start_sequence_monitoring(self):
        handler = self.SequenceHandler(self)
        self.sequence_observer = Observer()
        self.sequence_observer.schedule(handler, self.config['paths']['data_dir'])
        self.sequence_observer.start()

    def start_plate_monitoring(self):
        """Start monitoring plate_data.csv for changes"""
        if self.plate_observer:
            self.plate_observer.stop()
            self.plate_observer.join()

        handler = PlateDataHandler(
            self.config['paths']['data_dir'],
            self.config['files']['plate_data'],
            self.config['files']['processed_data'],
            controller=self
        )
        
        self.plate_observer = Observer()
        # Define the specific subdirectory to monitor for raw_plate_data.csv
        plate_data_watch_path = os.path.join(self.config['paths']['data_dir'], 'raw_plate_data')
        # Ensure the directory exists before starting the observer
        os.makedirs(plate_data_watch_path, exist_ok=True)
        self.plate_observer.schedule(handler, plate_data_watch_path, recursive=False)
        self.plate_observer.start()
        self.logger.info(f"Started plate data monitoring in: {plate_data_watch_path}")

    def start_evagreen_monitoring(self):
        """Start monitoring for raw_evagreen_data.csv in its subdirectory"""
        if self.evagreen_observer:
            self.evagreen_observer.stop()
            self.evagreen_observer.join()

        handler = EvagreenHandler(self)
        self.evagreen_observer = Observer()
        
        # Define the specific subdirectory to monitor for raw_evagreen_data.csv
        evagreen_watch_path = os.path.join(self.config['paths']['data_dir'], 'raw_evagreen_data')
        # Ensure the directory exists before starting the observer
        os.makedirs(evagreen_watch_path, exist_ok=True)
        self.evagreen_observer.schedule(handler, evagreen_watch_path, recursive=False)
        self.evagreen_observer.start()
        self.logger.info(f"Started evagreen monitoring in: {evagreen_watch_path}")

    def generate_lab_files(self):
        """Generate worklist and plate layout files"""
        try:
            # Generate pipetting worklist
            python_path = sys.executable
            subprocess.run([
                python_path, 'seq_to_pipetting_steps.py',
                os.path.join(self.config['paths']['data_dir'], 'sequence_query.txt'),
                os.path.join(self.config['paths']['data_dir'], 'sequence_segments.csv'), 
                self.config['paths']['worklists_dir']
            ], check=True)


            # Volume tracking disabled for now
            # inventory_dir = os.path.join(self.config['paths']['inventory_dir'])
            # dna_tracker = DNATracker(inventory_dir)
            # rgnt_tracker = RgntTracker(inventory_dir)
            # updates_dna = dna_tracker.update_volumes('worklists/fragment_assembly_worklist.csv')
            # worklist_files = [
            #     'worklists/GGMM_Wklist.csv',
            #     'worklists/PrimerTransfer_Wklist.csv',
            #     'worklists/PCRMM_Transfer_Wklist.csv',
            #     'worklists/PCR_product_dilution_Wklist.csv',
            #     'worklists/TXTL_Wklist.csv'
            # ]
            # rgnt_updates = []
            # for worklist_file in worklist_files:
            #     updates = rgnt_tracker.update_volumes(worklist_file)
            #     rgnt_updates.extend(updates)
            # self.logger.info("Updated DNA volumes")
            # for update in updates_dna:
            #     self.logger.info(
            #         f"Fragment {update['fragment']} in well {update['well']}: "
            #         f"used {update['volume_used']}µL, {update['volume_remaining']}µL remaining"
            #     )
            # self.logger.info("Updated reagent volumes")
            # for update in rgnt_updates:
            #     self.logger.info(
            #         f"Reagent {update['reagent']} in well {update['well']}: "
            #         f"used {update['volume_used']}µL, {update['volume_remaining']}µL remaining"
            #     )
            
            # Generate plate layout
            python_path = sys.executable
            subprocess.run([
                python_path, 'generate_assay_plate.py',
                os.path.join(self.config['paths']['data_dir'], 'sequence_query.txt'),
                os.path.join(self.config['paths']['data_dir'], 'sequence_segments.csv'), 
                'plate_layout.json'
            ], check=True)
            
            self.logger.info("Generated worklist and plate layout")
            self.sequence_observer.stop()
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating lab files: {e}")
            return False
        
    def run(self):
        self.logger.info("Starting lab automation system")
        try:
            self.start_sequence_monitoring()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            if self.sequence_observer:
                self.sequence_observer.stop()
                self.sequence_observer.join()
            if self.evagreen_observer:  # Uncomment evagreen observer cleanup
                self.evagreen_observer.stop()
                self.evagreen_observer.join()
            if self.plate_observer:
                self.plate_observer.stop()
                self.plate_observer.join()
        except Exception as e:
            self.status = LabState.ERROR
            self.logger.error(f"Error: {e}")
            send_sms("AutoLab ERROR: system crash")
    
if __name__ == "__main__":
    config_path = os.path.join('configs', 'lab_config.yml')
    
    # Create default config if it doesn't exist
    if not os.path.exists(config_path):
        default_config = {
            'paths': {
                'data_dir': "data",
                'worklists_dir': "worklists",
                'inventory_dir': "inventories",
                'logs_dir': "logs"
            },
            'files': {
                'plate_data': "raw_plate_data.csv",
                'processed_data': "phenotype.json",
                'sequence_query': "sequence_query.txt",
                'sequence_segments': "sequence_segments.csv"
            },
            'sftp': {
                'hostname': os.environ.get('PRAXIS_GPU_HOST', 'your.gpu.server'),
                'username': os.environ.get('PRAXIS_GPU_USER', 'user'),
                'remote_path': os.environ.get('PRAXIS_GPU_REMOTE_PATH', '/path/to/agent/data/plate_data'),
                'key_filename': os.environ.get('PRAXIS_SSH_KEY', '~/.ssh/id_rsa'),
                'port': int(os.environ.get('PRAXIS_GPU_PORT', 22))
            }
        }
        
        # Create worklists directory if it doesn't exist
        os.makedirs(default_config['paths']['worklists_dir'], exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Created default config at {config_path}")
        print("Please edit the config file with your settings before running again.")
        exit(1)
    
    try:    
        automation = LabController(config_path)
        automation.run()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)