import os
import json
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd

class PhenotypeHandler(FileSystemEventHandler):
    """Handler for monitoring phenotype data files"""
    def __init__(self, phenotype_file):
        self.phenotype_file = os.path.abspath(phenotype_file)  # Use absolute path
        self.phenotype_data = None
        self.logger = logging.getLogger(__name__)
        
    def on_modified(self, event):
        if event.src_path == self.phenotype_file:
            # Add small delay to ensure file writing is complete
            time.sleep(0.1)
            try:
                with open(self.phenotype_file, 'r') as f:
                    self.logger.info(f"Reading phenotype file: {self.phenotype_file}")
                    content = f.read()
                    if content.strip():  # Check if file is not empty
                        self.phenotype_data = json.loads(content)
                        self.logger.info("Successfully loaded phenotype data")
                    else:
                        self.logger.warning("Phenotype file is empty")
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {str(e)}")
                self.logger.debug(f"Content that failed to parse: {content}")
            except Exception as e:
                self.logger.error(f"Error reading phenotype file: {str(e)}")

def monitor_phenotype_data(phenotype_file):
    """
    Monitor for phenotype data from the lab automation system.
    
    Args:
        phenotype_file (str): Path to the phenotype data JSON file
        
    Returns:
        tuple: (phenotype_data, invalid_sequences)
            - phenotype_data: dict mapping sequences to their measurements
            - invalid_sequences: list of sequences that were invalid
    """
    phenotype_file = os.path.abspath(phenotype_file)
    logger = logging.getLogger(__name__)
    
    # Check if file exists first
    if not os.path.exists(phenotype_file):
        print(f"Phenotype file {phenotype_file} does not exist yet. Waiting for it to be created...")
    
    # Setup monitoring
    handler = PhenotypeHandler(phenotype_file)
    observer = Observer()
    watch_dir = os.path.dirname(phenotype_file)
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    
    try:
        print(f"Monitoring {phenotype_file} for updates...")
        wait_time = 0
        
        while handler.phenotype_data is None:
            # Print status message every 10 minutes
            if wait_time % 600 == 0 and wait_time > 0:
                print(f"Still waiting for phenotype data... ({wait_time // 60} minutes)")
            
            time.sleep(1)
            wait_time += 1
            
        # Process the monitored data
        phenotype_data = {}
        invalid_sequences = []
        
        print(f"Received phenotype data after waiting {wait_time // 60} minutes")
        
        for seq, data in handler.phenotype_data.items():
            # Check valid flag explicitly first
            if not data.get('valid', True):
                invalid_sequences.append(seq)
                logger.warning(f"Sequence marked as invalid: {seq}")
                continue
                
            # Then check measurements  
            if (not isinstance(data.get('measurements', None), list) or
                len(data.get('measurements', [])) == 0 or
                not all(isinstance(m, (int, float)) for m in data['measurements']) or
                any(isinstance(m, bool) for m in data['measurements']) or
                any(pd.isna(m) for m in data['measurements'])):
                
                invalid_sequences.append(seq)
                logger.warning(f"Invalid measurements for sequence: {seq}")
                if 'measurements' in data:
                    logger.debug(f"Invalid measurements: {data['measurements']}")
                continue
                
            # If we get here, sequence is valid
            phenotype_data[seq] = data['measurements']
                
        print(f"Processed phenotype data: {len(phenotype_data)} valid sequences, {len(invalid_sequences)} invalid sequences")
        print(phenotype_data)
        return phenotype_data, invalid_sequences
        
    except Exception as e:
        logger.error(f"Error monitoring phenotype data: {e}")
        return {}, []
    finally:
        observer.stop()
        observer.join()