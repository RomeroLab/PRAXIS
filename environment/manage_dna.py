import argparse
from lab_controller import DNATracker

def main():
    parser = argparse.ArgumentParser(description='DNA Inventory Management System')
    
    # Add arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--refill', action='store_true', 
                      help='Refill all wells below threshold')
    group.add_argument('--well', 
                      help='Refill specific well (e.g., A01)')
    group.add_argument('--fragment', 
                      help='Refill specific fragment (e.g., p1f0)')
    
    parser.add_argument('--volume', type=float, default=200,
                       help='Volume to add in µL (default: 200)')
    parser.add_argument('--threshold', type=float, default=50,
                       help='Low volume threshold in µL (default: 50)')
    parser.add_argument('--report', action='store_true',
                       help='Print inventory report')
    
    args = parser.parse_args()
    tracker = DNATracker()
    
    try:
        if args.refill:
            print(f"Refilling all wells below {args.threshold}µL...")
            updates = tracker.bulk_refill(args.threshold)
            print(f"Refilled {len(updates)} wells")
            
        elif args.well:
            print(f"Refilling well {args.well} with {args.volume}µL...")
            tracker.refill_dna(well=args.well, volume_added=args.volume)
            
        elif args.fragment:
            print(f"Refilling fragment {args.fragment} with {args.volume}µL...")
            tracker.refill_dna(fragment_id=args.fragment, volume_added=args.volume)
        
        # Always show report at the end
        print("\nCurrent Inventory Status:")
        tracker.print_inventory_report()
        tracker.export_to_csv()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())