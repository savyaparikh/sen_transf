import time
import pandas as pd
import os

def read_live_msrbyield(file_path=r"C:\Users\savya\OneDrive\MSRB_Yield.xlsx", interval_sec=30):
    """
    Reads the live MSRB_Yield Excel file into a DataFrame every 30 seconds.
    Keeps printing the latest shape and last row to show updates.
    """

    if not os.path.exists(file_path):
        print(f"⚠️ File not found: {file_path}")
        return

    print(f"Watching {file_path} for updates. Refresh every {interval_sec} seconds. Ctrl+C to stop.")
    
    while True:
        try:
            df = pd.read_excel(file_path, sheet_name="Sheet1")
            print(f"\n✅ Loaded {len(df)} rows at {pd.Timestamp.now()}")
            print(df.tail(3))  # show the last 3 rows
        except Exception as e:
            print("⚠️ Error reading file (maybe syncing/locked):", e)
        
        time.sleep(interval_sec)

# Run like this:
read_live_msrbyield()
