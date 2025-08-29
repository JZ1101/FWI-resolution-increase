#!/usr/bin/env python3
"""
Monitor preprocessing progress
"""

import time
import subprocess
from pathlib import Path
import os

def get_file_size(filepath):
    """Get file size in MB"""
    if Path(filepath).exists():
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0

def check_process():
    """Check if preprocessing is still running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'run_preprocessing'], 
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def main():
    log_file = "preprocessing_full.log"
    output_file = "data/02_final_unified/unified_fwi_dataset_2010_2017.nc"
    
    print("📊 PREPROCESSING MONITOR")
    print("="*50)
    
    while True:
        # Check if process is running
        is_running = check_process()
        
        # Get last log lines
        if Path(log_file).exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                last_lines = lines[-5:] if len(lines) >= 5 else lines
        else:
            last_lines = ["Log file not found"]
        
        # Get output file size
        output_size = get_file_size(output_file)
        
        # Clear screen and print status
        os.system('clear' if os.name == 'posix' else 'cls')
        print("📊 PREPROCESSING MONITOR")
        print("="*50)
        print(f"Status: {'🟢 RUNNING' if is_running else '🔴 STOPPED'}")
        print(f"Output file size: {output_size:.1f} MB")
        print("\nLast log entries:")
        print("-"*50)
        for line in last_lines:
            print(line.strip())
        print("-"*50)
        
        if not is_running and "COMPLETE" in ''.join(last_lines):
            print("\n✅ PREPROCESSING COMPLETE!")
            break
        
        time.sleep(5)

if __name__ == "__main__":
    main()