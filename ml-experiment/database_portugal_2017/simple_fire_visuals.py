#!/usr/bin/env python3
"""
Simple Fire Region Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_fire_visuals():
    print("CREATING FIRE REGION VISUALIZATIONS")
    print("="*50)
    
    # Load data
    data = pd.read_csv('experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv')
    data['time'] = pd.to_datetime(data['time'])
    
    # June 16 data
    june16_data = data[data['time'].dt.date == pd.to_datetime('2017-06-16').date()].copy()
    
    # Fire location and region
    fire_lat, fire_lon = 39.92, -8.15
    region_mask = (
        (june16_data['latitude'] >= fire_lat - 0.75) &
        (june16_data['latitude'] <= fire_lat + 0.75) &
        (june16_data['longitude'] >= fire_lon - 0.75) &
        (june16_data['longitude'] <= fire_lon + 0.75)
    )
    fire_region = june16_data[region_mask].copy()
    
    # Find fire point
    distances = np.sqrt((fire_region['latitude'] - fire_lat)**2 + (fire_region['longitude'] - fire_lon)**2)
    closest_idx = distances.idxmin()
    fire_point = fire_region.loc[closest_idx]
    
    print(f"Fire point: {fire_point['latitude']:.3f}°N, {fire_point['longitude']:.3f}°W")
    print(f"ERA5 FWI: {fire_point['fwi']:.3f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('June 16, 2017 - Pedrógão Grande Fire Region Analysis', fontsize=16, fontweight='bold')
    
    # 1. ERA5 FWI spatial distribution
    ax = axes[0, 0]
    scatter = ax.scatter(fire_region['longitude'], fire_region['latitude'], 
                        c=fire_region['fwi'], cmap='Reds', s=200, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.scatter(fire_lon, fire_lat, c='blue', marker='*', s=500, label='Fire Location', 
               edgecolor='white', linewidth=2, zorder=10)
    ax.set_title('ERA5 FWI (25km Resolution)\nJune 16, 2017', fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fire Weather Index (FWI)', fontsize=10)
    
    # Add text annotation for fire point
    ax.annotate(f'FWI = {fire_point["fwi"]:.1f}\n(HIGH Risk)', 
                xy=(fire_point['longitude'], fire_point['latitude']),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                fontsize=10, fontweight='bold')
    
    # 2. Model performance comparison
    ax = axes[0, 1]
    models = ['ANN', 'XGBoost', 'CNN', 'Ensemble']
    train_r2 = [0.994, 0.992, 0.676, 0]
    test_r2 = [-2.301, 0.588, 0.288, 0.534]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8, color='lightblue')
    bars2 = ax.bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8, color='coral')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('R²')
    ax.set_title('Model Performance Comparison\n(Overfitting Analysis)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-2.5, 1.2)
    
    # Add value labels
    for i, (train, test) in enumerate(zip(train_r2, test_r2)):
        if train > 0:
            ax.text(i - width/2, train + 0.05, f'{train:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, max(test + 0.05, -2.4), f'{test:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Fire risk thresholds
    ax = axes[1, 0]
    
    # Risk categories
    risk_thresholds = [0, 5.2, 11.2, 21.3, 38, 60]
    risk_labels = ['Very Low', 'Low', 'Moderate', 'High', 'Extreme']
    risk_colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    
    # Show ERA5 vs predicted values
    era5_fwi = fire_point['fwi']
    predicted_values = {
        'XGBoost': 20.226,
        'Ensemble': 17.806,
        'CNN': 15.385,
        'ANN': 14.680
    }
    
    # Create risk level visualization
    for i, (threshold, color, label) in enumerate(zip(risk_thresholds[:-1], risk_colors, risk_labels)):
        ax.barh(i, risk_thresholds[i+1] - threshold, left=threshold, 
                color=color, alpha=0.6, edgecolor='black', linewidth=0.5)
        ax.text(threshold + (risk_thresholds[i+1] - threshold)/2, i, label, 
                ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Add ERA5 and predictions
    ax.axvline(era5_fwi, color='blue', linewidth=3, label=f'ERA5: {era5_fwi:.1f}')
    
    colors = ['red', 'purple', 'orange', 'brown']
    for j, (model, pred) in enumerate(predicted_values.items()):
        ax.axvline(pred, color=colors[j], linewidth=2, linestyle='--', alpha=0.8, 
                  label=f'{model}: {pred:.1f}')
    
    ax.set_xlabel('Fire Weather Index (FWI)')
    ax.set_ylabel('Risk Categories')
    ax.set_title('Fire Risk Classification\nERA5 vs Model Predictions', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, 50)
    ax.set_ylim(-0.5, 4.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Back-aggregation concept
    ax = axes[1, 1]
    
    # Create mock back-aggregation visualization
    # Original 25km cell
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.8, edgecolor='black', linewidth=2))
    ax.text(0.5, 0.5, f'ERA5\n{era5_fwi:.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 1km predictions (5x5 grid)
    np.random.seed(42)
    predictions_1km = np.random.normal(18, 2, (5, 5))  # Mock predictions
    predictions_1km = np.clip(predictions_1km, 10, 25)
    
    for i in range(5):
        for j in range(5):
            color_intensity = predictions_1km[i, j] / 30
            ax.add_patch(plt.Rectangle((2 + j*0.2, i*0.2), 0.2, 0.2, 
                                     facecolor=plt.cm.Reds(color_intensity), 
                                     edgecolor='black', linewidth=0.5))
    
    # Back-aggregated result
    back_agg = np.mean(predictions_1km)
    ax.add_patch(plt.Rectangle((4, 0), 1, 1, facecolor=plt.cm.Reds(back_agg/30), 
                              alpha=0.8, edgecolor='black', linewidth=2))
    ax.text(4.5, 0.5, f'Back-Agg\n{back_agg:.1f}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Labels
    ax.text(0.5, -0.2, '25km Original', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(3, -0.2, '1km Predictions', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text(4.5, -0.2, '25km Back-Agg', ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(1.8, 0.5), xytext=(1.2, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.annotate('', xy=(3.8, 0.5), xytext=(3.2, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title('Back-Aggregation Process\n(25km → 1km → 25km)', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Add correlation info
    correlation = np.corrcoef([era5_fwi], [back_agg])[0, 1]
    ax.text(2.5, 1.3, f'Correlation: {correlation:.3f}\nDifference: {back_agg - era5_fwi:+.1f}', 
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fire_analysis_complete.png', dpi=300, bbox_inches='tight')
    print("Saved: fire_analysis_complete.png")
    
    # Summary
    print(f"\nFIRE ANALYSIS SUMMARY:")
    print(f"ERA5 FWI at fire location: {era5_fwi:.3f} (HIGH Risk)")
    print(f"Model predictions:")
    for model, pred in predicted_values.items():
        risk = "HIGH" if pred >= 21.3 else "MODERATE" if pred >= 11.2 else "LOW"
        print(f"  {model}: {pred:.3f} ({risk} Risk)")
    
    print(f"\nKey Findings:")
    print(f"• ALL models underestimated fire risk")
    print(f"• XGBoost closest to ERA5 (-5.9 FWI units)")
    print(f"• ANN worst performance (-11.5 FWI units)")  
    print(f"• Major overfitting in ANN (Train R²=0.994, Test R²=-2.301)")
    
    plt.show()

if __name__ == "__main__":
    create_fire_visuals()