from lab_controller import RgntTracker

# Create a reagent tracker instance
reagent_tracker = RgntTracker()

# Refill a reagent by ID
reagent_tracker.refill_reagent(reagent_id="GGMM", volume_added=1000)


# Check inventory
reagent_tracker.print_inventory_report()