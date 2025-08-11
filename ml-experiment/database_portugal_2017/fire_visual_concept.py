#!/usr/bin/env python3
"""
Fire Location Visual Concept - What you requested
"""

import numpy as np
import matplotlib.pyplot as plt

def create_fire_visual_concept():
    """Create the concept visuals you described"""
    
    print("CREATING FIRE LOCATION VISUALIZATION CONCEPT")
    print("="*60)
    
    # Fire location
    fire_lat, fire_lon = 39.92, -8.15
    
    # ====== SET 1: 25km Resolution Comparison ======
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Set 1: 25km Resolution - ERA5 vs Back-aggregated\nJune 16, 2017 Pedrógão Grande Fire', 
                 fontsize=16, fontweight='bold')
    
    # Create mock 25km grid (5x5 grid around fire)
    np.random.seed(42)
    lats_25km = np.arange(39.0, 41.1, 0.25)  # 25km ≈ 0.25°
    lons_25km = np.arange(-9.0, -7.1, 0.25)  # 25km ≈ 0.25°
    
    # Mock ERA5 FWI data (realistic values)
    era5_fwi = np.array([
        [15, 18, 22, 28, 25],
        [17, 21, 25, 32, 28],
        [20, 24, 26, 35, 30],  # Fire location in this row
        [18, 22, 28, 30, 26],
        [16, 19, 24, 27, 23]
    ])
    
    # Mock back-aggregated data (underestimated)
    xgb_back = era5_fwi * 0.77  # XGBoost back-agg
    ann_back = era5_fwi * 0.56  # ANN back-agg
    ens_back = era5_fwi * 0.68  # Ensemble back-agg
    
    # 1. Original ERA5 FWI (25km)
    ax = axes[0, 0]
    im = ax.imshow(era5_fwi, cmap='Reds', alpha=0.8, extent=[-9, -7, 39, 41])
    
    # Add fire location
    ax.scatter(fire_lon, fire_lat, c='blue', marker='*', s=500, 
               edgecolor='white', linewidth=2, zorder=10, label='Fire Location')
    
    # Add grid values
    for i in range(5):
        for j in range(5):
            ax.text(lons_25km[j], lats_25km[i], f'{era5_fwi[4-i, j]:.0f}', 
                   ha='center', va='center', fontweight='bold', fontsize=10,
                   color='white' if era5_fwi[4-i, j] > 25 else 'black')
    
    ax.set_title('Original ERA5 FWI (25km)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='FWI')
    
    # 2. XGBoost Back-aggregated
    ax = axes[0, 1]
    im = ax.imshow(xgb_back, cmap='Reds', alpha=0.8, extent=[-9, -7, 39, 41])
    ax.scatter(fire_lon, fire_lat, c='blue', marker='*', s=500, 
               edgecolor='white', linewidth=2, zorder=10, label='Fire Location')
    
    for i in range(5):
        for j in range(5):
            ax.text(lons_25km[j], lats_25km[i], f'{xgb_back[4-i, j]:.0f}', 
                   ha='center', va='center', fontweight='bold', fontsize=10,
                   color='white' if xgb_back[4-i, j] > 20 else 'black')
    
    ax.set_title('XGBoost Back-aggregated (25km)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='FWI')
    
    # 3. ANN Back-aggregated
    ax = axes[1, 0]
    im = ax.imshow(ann_back, cmap='Reds', alpha=0.8, extent=[-9, -7, 39, 41])
    ax.scatter(fire_lon, fire_lat, c='blue', marker='*', s=500, 
               edgecolor='white', linewidth=2, zorder=10, label='Fire Location')
    
    for i in range(5):
        for j in range(5):
            ax.text(lons_25km[j], lats_25km[i], f'{ann_back[4-i, j]:.0f}', 
                   ha='center', va='center', fontweight='bold', fontsize=10,
                   color='white' if ann_back[4-i, j] > 15 else 'black')
    
    ax.set_title('ANN Back-aggregated (25km)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='FWI')
    
    # 4. Ensemble Back-aggregated
    ax = axes[1, 1]
    im = ax.imshow(ens_back, cmap='Reds', alpha=0.8, extent=[-9, -7, 39, 41])
    ax.scatter(fire_lon, fire_lat, c='blue', marker='*', s=500, 
               edgecolor='white', linewidth=2, zorder=10, label='Fire Location')
    
    for i in range(5):
        for j in range(5):
            ax.text(lons_25km[j], lats_25km[i], f'{ens_back[4-i, j]:.0f}', 
                   ha='center', va='center', fontweight='bold', fontsize=10,
                   color='white' if ens_back[4-i, j] > 18 else 'black')
    
    ax.set_title('Ensemble Back-aggregated (25km)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='FWI')
    
    plt.tight_layout()
    plt.savefig('set1_25km_comparison_concept.png', dpi=300, bbox_inches='tight')
    print("Saved: set1_25km_comparison_concept.png")
    
    # Print comparison at fire location
    fire_i, fire_j = 2, 2  # Fire location in grid
    print(f"\nSet 1 - Fire Location Values (25km grid):")
    print(f"ERA5 Original: {era5_fwi[fire_i, fire_j]:.1f}")
    print(f"XGBoost Back-agg: {xgb_back[fire_i, fire_j]:.1f} (diff: {xgb_back[fire_i, fire_j] - era5_fwi[fire_i, fire_j]:+.1f})")
    print(f"ANN Back-agg: {ann_back[fire_i, fire_j]:.1f} (diff: {ann_back[fire_i, fire_j] - era5_fwi[fire_i, fire_j]:+.1f})")
    print(f"Ensemble Back-agg: {ens_back[fire_i, fire_j]:.1f} (diff: {ens_back[fire_i, fire_j] - era5_fwi[fire_i, fire_j]:+.1f})")
    
    # ====== SET 2: 1km Resolution Predictions ======
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Set 2: 1km Resolution Predictions - Fire Location\nJune 16, 2017 Pedrógão Grande Fire', 
                 fontsize=16, fontweight='bold')
    
    # Create 1km grid (25x25 grid in one 25km cell)
    grid_size = 25
    x_1km = np.linspace(-8.375, -8.125, grid_size)  # 1km resolution
    y_1km = np.linspace(39.775, 40.025, grid_size)   # 1km resolution
    
    # Generate realistic 1km predictions
    np.random.seed(42)
    
    # Create distance-based patterns around fire location
    X_grid, Y_grid = np.meshgrid(x_1km, y_1km)
    dist_from_fire = np.sqrt((X_grid - fire_lon)**2 + (Y_grid - fire_lat)**2)
    
    # Base pattern with some variation
    base_pattern = 20 + 5 * np.exp(-dist_from_fire * 50)  # Higher values closer to fire
    
    # Model-specific variations
    xgb_1km = base_pattern + np.random.normal(0, 1.5, (grid_size, grid_size))
    xgb_1km = np.clip(xgb_1km, 10, 35)
    
    ann_1km = base_pattern * 0.7 + np.random.normal(0, 0.8, (grid_size, grid_size))  # ANN underestimates more
    ann_1km = np.clip(ann_1km, 8, 25)
    
    cnn_1km = base_pattern * 0.8 + np.random.normal(0, 1.2, (grid_size, grid_size))
    cnn_1km = np.clip(cnn_1km, 12, 30)
    
    ens_1km = (xgb_1km + cnn_1km) / 2  # Ensemble
    
    extent = [x_1km.min(), x_1km.max(), y_1km.min(), y_1km.max()]
    
    # 1. XGBoost 1km predictions
    ax = axes[0, 0]
    im = ax.imshow(xgb_1km, cmap='Reds', alpha=0.8, extent=extent, origin='lower')
    ax.scatter(fire_lon, fire_lat, c='blue', marker='*', s=500, 
               edgecolor='white', linewidth=2, zorder=10, label='Fire Location')
    ax.set_title('XGBoost Predictions (1km)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Predicted FWI')
    
    # 2. ANN 1km predictions
    ax = axes[0, 1]
    im = ax.imshow(ann_1km, cmap='Reds', alpha=0.8, extent=extent, origin='lower')
    ax.scatter(fire_lon, fire_lat, c='blue', marker='*', s=500, 
               edgecolor='white', linewidth=2, zorder=10, label='Fire Location')
    ax.set_title('ANN Predictions (1km)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Predicted FWI')
    
    # 3. Ensemble 1km predictions
    ax = axes[1, 0]
    im = ax.imshow(ens_1km, cmap='Reds', alpha=0.8, extent=extent, origin='lower')
    ax.scatter(fire_lon, fire_lat, c='blue', marker='*', s=500, 
               edgecolor='white', linewidth=2, zorder=10, label='Fire Location')
    ax.set_title('Ensemble Predictions (1km)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Predicted FWI')
    
    # 4. Statistics comparison
    ax = axes[1, 1]
    models = ['XGBoost', 'ANN', 'Ensemble']
    means = [xgb_1km.mean(), ann_1km.mean(), ens_1km.mean()]
    stds = [xgb_1km.std(), ann_1km.std(), ens_1km.std()]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, means, width, label='Mean FWI', alpha=0.8, color='red')
    bars2 = ax.bar(x + width/2, stds, width, label='Std Dev', alpha=0.8, color='blue')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('FWI Value')
    ax.set_title('1km Prediction Statistics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i - width/2, mean + 0.5, f'{mean:.1f}', ha='center', va='bottom', fontsize=10)
        ax.text(i + width/2, std + 0.1, f'{std:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('set2_1km_predictions_concept.png', dpi=300, bbox_inches='tight')
    print("Saved: set2_1km_predictions_concept.png")
    
    # Find fire location values in 1km predictions
    fire_idx = grid_size // 2  # Center of grid
    print(f"\nSet 2 - Fire Location Predictions (1km):")
    print(f"XGBoost: {xgb_1km[fire_idx, fire_idx]:.1f}")
    print(f"ANN: {ann_1km[fire_idx, fire_idx]:.1f}")
    print(f"Ensemble: {ens_1km[fire_idx, fire_idx]:.1f}")
    
    print(f"\nRegion statistics (1km predictions):")
    print(f"XGBoost: mean={xgb_1km.mean():.1f}, std={xgb_1km.std():.1f}")
    print(f"ANN: mean={ann_1km.mean():.1f}, std={ann_1km.std():.1f}")
    print(f"Ensemble: mean={ens_1km.mean():.1f}, std={ens_1km.std():.1f}")
    
    print(f"\n" + "="*60)
    print("FIRE LOCATION VISUALIZATION CONCEPT COMPLETED")
    print("="*60)
    print("✅ Set 1: 25km ERA5 vs Back-aggregated comparison")
    print("✅ Set 2: 1km resolution predictions by all models")
    print("\nKey Insight: Set 1 shows back-aggregation fails to match ERA5")
    print("Key Insight: Set 2 shows 1km predictions vary by model but all underestimate")

if __name__ == "__main__":
    create_fire_visual_concept()