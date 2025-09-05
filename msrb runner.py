import threading
import time
from msrb_live import live_msrbyield_to_excel
from msrb_reader import read_live_msrbyield

# Wrapped functions with better print prefixes
def writer_task():
    print("🟢 [Writer-T1] Started appending live MSRB_Yield data to Excel...")
    live_msrbyield_to_excel()

def reader_task():
    print("🔵 [Reader-T2] Started reading MSRB_Yield Excel into DataFrame...")
    read_live_msrbyield()

if __name__ == "__main__":
    # Writer thread
    t1 = threading.Thread(target=writer_task, daemon=True, name="Writer-T1")

    # Reader thread
    t2 = threading.Thread(target=reader_task, daemon=True, name="Reader-T2")

    # Start both
    t1.start()
    t2.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping Writer & Reader threads.")
