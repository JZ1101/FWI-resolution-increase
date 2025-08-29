#!/usr/bin/env python3
"""
Step 1: Create master 1km grid for Portugal.
Creates the spatial reference grid that all data will be mapped to.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import load_config, create_master_grid
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Create and save master grid."""
    logger.info("="*60)
    logger.info("STEP 1: CREATE MASTER GRID")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    output_path = Path(config['paths']['master_grid'])
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create master grid
    logger.info("Creating master grid for Portugal...")
    logger.info(f"  Bounds: {config['geography']['lat_min']}°N to {config['geography']['lat_max']}°N")
    logger.info(f"         {config['geography']['lon_min']}°E to {config['geography']['lon_max']}°E")
    logger.info(f"  Resolution: {config['geography']['target_resolution']}° (~1km)")
    
    master_grid = create_master_grid(config)
    
    # Log grid info
    logger.info(f"Grid dimensions: {master_grid.dims}")
    logger.info(f"Variables: {list(master_grid.data_vars)}")
    
    # Save grid
    master_grid.to_netcdf(output_path)
    logger.info(f"✓ Saved master grid to: {output_path}")
    
    # Verification
    logger.info("\nVerification:")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    logger.info(f"  Grid points: {len(master_grid.latitude)} x {len(master_grid.longitude)}")
    
    return output_path


if __name__ == "__main__":
    output = main()
    print(f"\n✓ Output: {output}")
    print("\nReady for next step: 02_process_fwi.py")