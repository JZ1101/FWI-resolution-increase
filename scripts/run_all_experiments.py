#!/usr/bin/env python3
"""Run all downscaling experiments"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import subprocess
import logging
from src.utils import setup_logging, Timer
import pandas as pd

logger = logging.getLogger(__name__)


def run_experiment(method_dir: Path) -> dict:
    """Run a single experiment"""
    
    run_script = method_dir / "run.py"
    if not run_script.exists():
        logger.warning(f"No run.py found in {method_dir}")
        return None
    
    logger.info(f"Running {method_dir.name}...")
    
    try:
        with Timer(f"Experiment {method_dir.name}"):
            result = subprocess.run(
                [sys.executable, str(run_script)],
                cwd=method_dir,
                capture_output=True,
                text=True
            )
            
        if result.returncode != 0:
            logger.error(f"Experiment failed: {result.stderr}")
            return {"status": "failed", "error": result.stderr}
        
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        return {"status": "error", "error": str(e)}


def main():
    setup_logging(level="INFO")
    
    experiments_dir = Path("experiments")
    methods = [
        "method1_interpolation",
        "method2_ml_terrain", 
        "method3_physics_ml",
        "method4_transformers"
    ]
    
    results = {}
    
    for method in methods:
        method_dir = experiments_dir / method
        if method_dir.exists():
            result = run_experiment(method_dir)
            results[method] = result
        else:
            logger.warning(f"Method directory not found: {method_dir}")
    
    logger.info("\n=== Experiment Summary ===")
    for method, result in results.items():
        if result:
            status = result.get("status", "unknown")
            logger.info(f"{method}: {status}")
    
    df = pd.DataFrame(results).T
    df.to_csv("outputs/experiment_summary.csv")
    logger.info("Summary saved to outputs/experiment_summary.csv")


if __name__ == "__main__":
    main()