#!/usr/bin/env python3
"""
Simple Fire Location Visual - Using existing results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_simple_fire_visual():
    print("CREATING SIMPLE FIRE LOCATION VISUAL")
    print("=" * 50)
    
    # Load existing data
    try:
        data = pd.read_csv('database_portugal_2017/experiment_2017_portugal/features_2017_COMPLETE_FINAL.csv')
        data['time'] = pd.to_datetime(data['time'])
    except:
        print("Could not load data file")
        return
    
    # Fire location and known results
    actual_fire_lat, actual_fire_lon = 39.92, -8.15
    era5_fwi = 26.168  # From previous runs
    
    # Known model predictions (from extract_june16_predictions.py)
    model_predictions = {
        'ERA5': 26.168,
        'XGBoost': 20.226,
        'ANN': 14.680,
        'CNN': 18.558,
        'Ensemble': 19.392
    }
    
    # Find nearest ERA5 point
    june16_data = data[data['time'].dt.date == pd.to_datetime('2017-06-16').date()].copy()
    distances = np.sqrt((june16_data['latitude'] - actual_fire_lat)**2 + 
                       (june16_data['longitude'] - actual_fire_lon)**2)
    nearest_era5_idx = distances.idxmin()
    nearest_era5_point = june16_data.loc[nearest_era5_idx]
    
    era5_lat = nearest_era5_point['latitude']
    era5_lon = nearest_era5_point['longitude']
    distance_km = distances.min() * 111  # Convert degrees to km
    
    print(f"Actual fire: {actual_fire_lat}°N, {actual_fire_lon}°W")
    print(f"ERA5 point: {era5_lat}°N, {era5_lon}°W")
    print(f"Distance: {distance_km:.1f} km")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('June 16, 2017 Pedrógão Grande Fire Analysis', fontsize=16, fontweight='bold')
    
    # GRAPH 1: Location comparison
    ax1.set_title('Graph 1: Fire Location vs ERA5 Grid Point', fontsize=14, fontweight='bold')
    
    # Plot fire location
    ax1.scatter(actual_fire_lon, actual_fire_lat, c='red', marker='*', s=800, 
               edgecolor='black', linewidth=3, label='Actual Fire Location', zorder=10)
    
    # Plot ERA5 point
    ax1.scatter(era5_lon, era5_lat, c='blue', marker='s', s=400, 
               edgecolor='black', linewidth=2, label=f'ERA5 Grid Point\\n(FWI = {era5_fwi:.1f})', zorder=9)
    
    # Connection line
    ax1.plot([actual_fire_lon, era5_lon], [actual_fire_lat, era5_lat], 
             'k--', alpha=0.7, linewidth=2, label=f'Distance: {distance_km:.1f} km')
    
    # Add region context
    region_data = june16_data[
        (june16_data['latitude'] >= actual_fire_lat - 0.5) &
        (june16_data['latitude'] <= actual_fire_lat + 0.5) &
        (june16_data['longitude'] >= actual_fire_lon - 0.5) &
        (june16_data['longitude'] <= actual_fire_lon + 0.5)
    ]
    
    scatter = ax1.scatter(region_data['longitude'], region_data['latitude'], 
                         c=region_data['fwi'], cmap='Reds', s=100, alpha=0.6, 
                         edgecolor='gray', linewidth=0.5)
    
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('ERA5 FWI', fontsize=10)
    
    # GRAPH 2: Model comparison
    ax2.set_title('Graph 2: ERA5 vs Model Predictions at Fire Location', fontsize=14, fontweight='bold')
    
    # Risk level background
    risk_levels = [
        ('Very Low', 0, 5.2, '#90EE90'),
        ('Low', 5.2, 11.2, '#FFFF99'),
        ('Moderate', 11.2, 21.3, '#FFB347'),
        ('High', 21.3, 38, '#FF6347'),
        ('Extreme', 38, 50, '#8B0000')
    ]
    
    for name, low, high, color in risk_levels:
        ax2.axhspan(low, high, alpha=0.3, color=color)
    
    # Model predictions
    models = list(model_predictions.keys())
    predictions = list(model_predictions.values())
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    bars = ax2.bar(models, predictions, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, pred in zip(bars, predictions):
        height = bar.get_height()
        # Determine risk level
        if pred >= 21.3:
            risk = "HIGH"
        elif pred >= 11.2:
            risk = "MODERATE"
        else:
            risk = "LOW"
        
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                f'{pred:.1f}\\n({risk})', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax2.set_ylabel('Fire Weather Index (FWI)')
    ax2.set_ylim(0, 35)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add annotation
    ax2.text(0.02, 0.98, f'Fire Location: {actual_fire_lat}°N, {actual_fire_lon}°W\\nDate: June 16, 2017\\nActual Fire: Pedrógão Grande',
             transform=ax2.transAxes, va='top', ha='left', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('fire_location_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: fire_location_comparison.png")
    
    # Summary table
    print(f"\\nRESULTS SUMMARY:")
    print("=" * 40)
    print(f"{'Model':<10} {'FWI':<8} {'Risk Level':<10} {'Difference':<10}")
    print("-" * 40)
    
    era5_val = model_predictions['ERA5']
    for model, pred in model_predictions.items():
        if pred >= 21.3:
            risk = "HIGH"
        elif pred >= 11.2:
            risk = "MODERATE" 
        else:
            risk = "LOW"
        
        diff = pred - era5_val if model != 'ERA5' else 0.0
        print(f"{model:<10} {pred:<8.1f} {risk:<10} {diff:+.1f}")
    
    print("\\nKEY FINDINGS:")
    print("• ALL models underestimated fire risk")
    print("• XGBoost closest to ERA5 (-5.9 FWI)")
    print("• ANN worst performance (-11.5 FWI)")
    print("• Fire occurred in HIGH risk conditions (FWI > 21)")
    
    plt.show()

if __name__ == "__main__":
    create_simple_fire_visual()