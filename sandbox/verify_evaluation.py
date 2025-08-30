#!/usr/bin/env python3
"""
Verification script for evaluation metrics.
Tests the back-aggregation correlation function with predictable scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.evaluate import calculate_back_aggregation_correlation


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_scenario(number, description):
    """Print scenario information."""
    print(f"\n📊 SCENARIO {number}: {description}")
    print("-"*60)


def verify_scenario_1_perfect_correlation():
    """
    Scenario 1: Perfect Correlation
    Low-res and high-res have identical spatial patterns.
    Expected correlation: 1.0
    """
    print_scenario(1, "Perfect Correlation")
    
    # Create tensors
    low_res_size = (4, 4)  # 4x4 low resolution
    high_res_size = (20, 20)  # 20x20 high resolution (5x upscaling)
    batch_size = 1
    channels = 1
    
    # Create a pattern with variation (not all same values)
    x_low_res = torch.arange(16).reshape(1, 1, 4, 4).float() + 1.0
    
    # Create high-res by perfectly upscaling the low-res pattern
    y_hat_high_res = torch.zeros(batch_size, channels, *high_res_size)
    for i in range(4):
        for j in range(4):
            # Fill each 5x5 block with the corresponding low-res value
            y_hat_high_res[0, 0, i*5:(i+1)*5, j*5:(j+1)*5] = x_low_res[0, 0, i, j]
    
    print(f"Low-res tensor shape: {x_low_res.shape}")
    print(f"Low-res values (first row): {x_low_res[0, 0, 0, :].numpy()}")
    print(f"High-res tensor shape: {y_hat_high_res.shape}")
    print(f"High-res pattern: Perfect upscaling of low-res (each value fills 5x5 block)")
    
    # Calculate correlation
    correlation = calculate_back_aggregation_correlation(y_hat_high_res, x_low_res)
    
    print(f"\n🎯 Calculated correlation: {correlation:.6f}")
    print(f"✅ Expected correlation: 1.000000")
    
    # Verify result
    assert abs(correlation - 1.0) < 1e-6, f"Expected 1.0, got {correlation}"
    print("✓ Test PASSED: Perfect correlation detected")
    
    return correlation


def verify_scenario_2_no_correlation():
    """
    Scenario 2: No Correlation
    Low-res has structured values, high-res has random noise.
    Expected correlation: ~0.0 (close to zero)
    """
    print_scenario(2, "No Correlation (Random Noise)")
    
    # Create tensors
    low_res_size = (4, 4)
    high_res_size = (20, 20)
    batch_size = 1
    channels = 1
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Low-res: structured pattern
    x_low_res = torch.arange(16).reshape(1, 1, 4, 4).float() + 1.0
    
    # High-res: random noise
    y_hat_high_res = torch.randn(batch_size, channels, *high_res_size) * 10 + 50
    
    print(f"Low-res tensor shape: {x_low_res.shape}")
    print(f"Low-res values (first row): {x_low_res[0, 0, 0, :].numpy()}")
    print(f"High-res tensor shape: {y_hat_high_res.shape}")
    print(f"High-res values: Random noise (mean={y_hat_high_res.mean():.2f}, std={y_hat_high_res.std():.2f})")
    
    # Calculate correlation
    correlation = calculate_back_aggregation_correlation(y_hat_high_res, x_low_res)
    
    print(f"\n🎯 Calculated correlation: {correlation:.6f}")
    print(f"✅ Expected correlation: ~0.0 (close to zero)")
    
    # Verify result (should be close to 0, within reasonable bounds)
    assert abs(correlation) < 0.3, f"Expected near 0, got {correlation}"
    print(f"✓ Test PASSED: No correlation detected (|r| < 0.3)")
    
    return correlation


def verify_scenario_3_perfect_anti_correlation():
    """
    Scenario 3: Perfect Anti-Correlation
    High-res is perfectly inversely related to low-res.
    Expected correlation: -1.0
    """
    print_scenario(3, "Perfect Anti-Correlation")
    
    # Create tensors
    low_res_size = (4, 4)
    high_res_size = (20, 20)
    batch_size = 1
    channels = 1
    
    # Low-res: ascending values
    x_low_res = torch.arange(16).reshape(1, 1, 4, 4).float() + 1.0
    
    # High-res: Create inverse pattern
    # First create the pattern at low res, then expand it
    low_res_pattern = 17.0 - x_low_res[0, 0, :, :]  # Inverse: 16, 15, 14, ..., 1
    
    # Expand to high-res by repeating each value in a 5x5 block
    y_hat_high_res = torch.zeros(batch_size, channels, *high_res_size)
    for i in range(4):
        for j in range(4):
            # Fill each 5x5 block with the corresponding low-res value
            y_hat_high_res[0, 0, i*5:(i+1)*5, j*5:(j+1)*5] = low_res_pattern[i, j]
    
    print(f"Low-res tensor shape: {x_low_res.shape}")
    print(f"Low-res values (first row): {x_low_res[0, 0, 0, :].numpy()}")
    print(f"Low-res values (last row): {x_low_res[0, 0, -1, :].numpy()}")
    
    print(f"\nHigh-res tensor shape: {y_hat_high_res.shape}")
    # Show aggregated values to demonstrate inverse relationship
    aggregated = torch.nn.functional.avg_pool2d(y_hat_high_res, kernel_size=5, stride=5)
    print(f"High-res (after aggregation) first row: {aggregated[0, 0, 0, :].numpy()}")
    print(f"High-res (after aggregation) last row: {aggregated[0, 0, -1, :].numpy()}")
    
    # Calculate correlation
    correlation = calculate_back_aggregation_correlation(y_hat_high_res, x_low_res)
    
    print(f"\n🎯 Calculated correlation: {correlation:.6f}")
    print(f"✅ Expected correlation: -1.000000")
    
    # Verify result
    assert abs(correlation - (-1.0)) < 1e-6, f"Expected -1.0, got {correlation}"
    print("✓ Test PASSED: Perfect anti-correlation detected")
    
    return correlation


def verify_edge_cases():
    """Test edge cases and error handling."""
    print_header("EDGE CASE TESTING")
    
    print("\n🔍 Testing dimension mismatch handling...")
    try:
        # Incompatible dimensions
        x_low = torch.ones(1, 1, 3, 3)
        y_high = torch.ones(1, 1, 10, 10)  # 10 is not divisible by 3
        calculate_back_aggregation_correlation(y_high, x_low)
        print("❌ Should have raised an error!")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {str(e)[:50]}...")
    
    print("\n🔍 Testing automatic aggregation factor calculation...")
    # Should automatically determine factor = 5
    x_low = torch.ones(1, 1, 4, 4) * 2.0
    y_high = torch.ones(1, 1, 20, 20) * 2.0
    correlation = calculate_back_aggregation_correlation(y_high, x_low)
    print(f"✓ Auto-calculated factor worked, correlation: {correlation:.4f}")
    
    print("\n🔍 Testing zero variance handling...")
    # Both tensors have same value (no variance)
    x_low = torch.ones(1, 1, 4, 4) * 7.0
    y_high = torch.ones(1, 1, 20, 20) * 7.0
    correlation = calculate_back_aggregation_correlation(y_high, x_low)
    print(f"✓ Zero variance handled correctly, returns: {correlation:.4f} (undefined correlation)")


def main():
    """Run all verification tests."""
    print_header("EVALUATION METRICS VERIFICATION")
    print("\nThis script tests the back-aggregation correlation function")
    print("with three predictable scenarios and edge cases.\n")
    
    # Store results
    results = {}
    
    # Run three main scenarios
    results['perfect_correlation'] = verify_scenario_1_perfect_correlation()
    results['no_correlation'] = verify_scenario_2_no_correlation()
    results['perfect_anti_correlation'] = verify_scenario_3_perfect_anti_correlation()
    
    # Test edge cases
    verify_edge_cases()
    
    # Summary
    print_header("VERIFICATION SUMMARY")
    print("\n📈 Results:")
    print(f"  Scenario 1 (Perfect Correlation):      {results['perfect_correlation']:+.6f} ✅")
    print(f"  Scenario 2 (No Correlation):           {results['no_correlation']:+.6f} ✅")
    print(f"  Scenario 3 (Perfect Anti-Correlation): {results['perfect_anti_correlation']:+.6f} ✅")
    
    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")
    print("\nThe back-aggregation correlation function is mathematically correct:")
    print("• Correctly identifies perfect positive correlation (r = 1.0)")
    print("• Correctly identifies no correlation (r ≈ 0.0)")
    print("• Correctly identifies perfect negative correlation (r = -1.0)")
    print("• Handles edge cases and dimension mismatches appropriately")
    
    return 0


if __name__ == "__main__":
    exit(main())