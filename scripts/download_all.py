#!/usr/bin/env python3
"""
Download all FWI-related datasets for years 2015-2017 (excluding 2018)
Runs all download scripts and organizes data in temp_database folder
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directories
SCRIPT_DIR = Path(__file__).parent
DOWNLOAD_DIR = SCRIPT_DIR / "download"
TEMP_DATABASE_DIR = SCRIPT_DIR / "temp_database"

# Download scripts to run (in order)
DOWNLOAD_SCRIPTS = [
    {
        "name": "ERA5 FWI Data",
        "path": DOWNLOAD_DIR / "ERA5_reanalysis_FWI" / "ERA5_25km_fwi.py",
        "output_dir": "ERA5_FWI"
    },
    {
        "name": "ERA5 Land Data", 
        "path": DOWNLOAD_DIR / "ERA5_reanalysis_ atmospheric_land-surface_parameters" / "ERA5_land.py",
        "output_dir": "ERA5_Land"
    },
    {
        "name": "ERA5 Daily Mean Atmospheric Data",
        "path": DOWNLOAD_DIR / "ERA5_reanalysis_atmospheric_parameters" / "ERA5_daily_mean.py", 
        "output_dir": "ERA5_Atmospheric"
    },
    {
        "name": "ERA5 Daily Temperature Data",
        "path": DOWNLOAD_DIR / "ERA5_reanalysis_atmospheric_parameters" / "ERA5_daily_temp.py",
        "output_dir": "ERA5_Temperature"
    },
    {
        "name": "UERRA Daily Data",
        "path": DOWNLOAD_DIR / "UERRA_reanalysis_atmospheric parameters" / "UERRA_daily.py",
        "output_dir": "UERRA_Daily"
    },
    {
        "name": "CMIP6 Daily Data",
        "path": DOWNLOAD_DIR / "CMIP6_modelled_data" / "CMIP6_daily.py",
        "output_dir": "CMIP6_Daily"
    }
]

def create_temp_database_structure():
    """Create the temp_database directory structure"""
    logger.info("Creating temp_database directory structure...")
    
    # Create main temp_database directory
    TEMP_DATABASE_DIR.mkdir(exist_ok=True)
    
    # Create subdirectories for each data type
    for script_info in DOWNLOAD_SCRIPTS:
        output_dir = TEMP_DATABASE_DIR / script_info["output_dir"]
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Created directory: {output_dir}")

def run_download_script(script_info):
    """Run a single download script"""
    script_name = script_info["name"]
    script_path = script_info["path"]
    output_dir = script_info["output_dir"]
    
    logger.info(f"Starting download: {script_name}")
    logger.info(f"Script: {script_path}")
    
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False
    
    try:
        # Change to script directory to run it
        script_dir = script_path.parent
        original_dir = os.getcwd()
        
        os.chdir(script_dir)
        
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path.name)], 
            capture_output=True, 
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        # Return to original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            logger.info(f"✅ Successfully completed: {script_name}")
            
            # Move downloaded files to temp_database
            move_downloaded_files(script_dir, output_dir, script_name)
            return True
        else:
            logger.error(f"❌ Failed: {script_name}")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ Timeout: {script_name} (1 hour limit exceeded)")
        os.chdir(original_dir)
        return False
    except Exception as e:
        logger.error(f"❌ Exception in {script_name}: {str(e)}")
        os.chdir(original_dir)
        return False

def move_downloaded_files(source_dir, output_subdir, script_name):
    """Move downloaded files from script directory to temp_database"""
    target_dir = TEMP_DATABASE_DIR / output_subdir
    
    # Common file extensions for downloaded data
    data_extensions = ['.nc', '.grib', '.grb', '.netcdf', '.hdf', '.h5']
    
    moved_files = 0
    
    for file_path in source_dir.glob('*'):
        if file_path.is_file():
            # Check if it's a data file
            if any(file_path.suffix.lower() == ext for ext in data_extensions):
                # Check if it's from our target years (2015-2017)
                if any(year in file_path.name for year in ['2015', '2016', '2017']):
                    try:
                        target_path = target_dir / file_path.name
                        shutil.move(str(file_path), str(target_path))
                        logger.info(f"Moved: {file_path.name} -> {output_subdir}/")
                        moved_files += 1
                    except Exception as e:
                        logger.error(f"Failed to move {file_path.name}: {str(e)}")
    
    logger.info(f"Moved {moved_files} files for {script_name}")

def check_downloads():
    """Check what files were downloaded"""
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    
    total_files = 0
    total_size = 0
    
    for script_info in DOWNLOAD_SCRIPTS:
        output_dir = TEMP_DATABASE_DIR / script_info["output_dir"]
        
        if output_dir.exists():
            files = list(output_dir.glob('*'))
            data_files = [f for f in files if f.is_file()]
            
            logger.info(f"\n{script_info['name']}:")
            logger.info(f"  Directory: {output_dir}")
            logger.info(f"  Files: {len(data_files)}")
            
            dir_size = 0
            for file_path in data_files:
                try:
                    size = file_path.stat().st_size
                    size_mb = size / (1024 * 1024)
                    dir_size += size
                    logger.info(f"    - {file_path.name} ({size_mb:.2f} MB)")
                except Exception as e:
                    logger.warning(f"    - {file_path.name} (size unknown)")
            
            total_files += len(data_files)
            total_size += dir_size
        else:
            logger.warning(f"\n{script_info['name']}: No directory found")
    
    total_size_gb = total_size / (1024 * 1024 * 1024)
    logger.info(f"\n📊 TOTAL SUMMARY:")
    logger.info(f"  Total files: {total_files}")
    logger.info(f"  Total size: {total_size_gb:.2f} GB")
    logger.info(f"  Storage location: {TEMP_DATABASE_DIR}")

def main():
    """Main function to run all downloads"""
    logger.info("🚀 Starting FWI Data Download Pipeline")
    logger.info("📅 Target years: 2015-2017 (excluding 2018)")
    logger.info("📁 Output directory: temp_database/")
    logger.info("="*60)
    
    # Create directory structure
    create_temp_database_structure()
    
    # Run each download script
    successful_downloads = 0
    failed_downloads = 0
    
    for i, script_info in enumerate(DOWNLOAD_SCRIPTS, 1):
        logger.info(f"\n📥 [{i}/{len(DOWNLOAD_SCRIPTS)}] {script_info['name']}")
        logger.info("-" * 40)
        
        success = run_download_script(script_info)
        
        if success:
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        logger.info(f"Progress: {i}/{len(DOWNLOAD_SCRIPTS)} scripts completed")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("🎯 PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"✅ Successful downloads: {successful_downloads}")
    logger.info(f"❌ Failed downloads: {failed_downloads}")
    
    if failed_downloads == 0:
        logger.info("🎉 All downloads completed successfully!")
    else:
        logger.warning(f"⚠️  {failed_downloads} downloads failed. Check logs above for details.")
    
    # Check and summarize what was downloaded
    check_downloads()
    
    logger.info(f"\n📁 All data stored in: {TEMP_DATABASE_DIR.absolute()}")

if __name__ == "__main__":
    main()