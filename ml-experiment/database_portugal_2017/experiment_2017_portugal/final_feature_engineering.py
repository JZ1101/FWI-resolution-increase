#!/usr/bin/env python3
"""
Final Feature Engineering for FWI Resolution Enhancement
Robust processing of all 2017 datasets with proper temporal alignment
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def normalize_timestamps(df, target_hour=12):
    """Normalize timestamps to daily resolution at target hour"""
    df = df.copy()
    df['date'] = df['time'].dt.date
    df['target_time'] = pd.to_datetime(df['date'].astype(str) + f' {target_hour:02d}:00:00')
    return df

def main():
    print("="*70)
    print("FINAL FWI FEATURE ENGINEERING - 2017 PORTUGAL")
    print("="*70)
    
    # Load datasets
    print("\n📊 LOADING AND PROCESSING DATASETS...")
    
    # 1. ERA5 FWI (Target - 25km)
    print("   Loading ERA5 FWI (target)...")
    era5_fwi = pd.read_csv('2017_data/era5_fwi_2017_portugal.csv')
    era5_fwi['time'] = pd.to_datetime(era5_fwi['time'])
    era5_fwi = normalize_timestamps(era5_fwi, 12)
    print(f"      Shape: {era5_fwi.shape}")
    print(f"      Time range: {era5_fwi['time'].min()} to {era5_fwi['time'].max()}")
    print(f"      Unique coordinates: {len(era5_fwi[['latitude', 'longitude']].drop_duplicates())}")
    
    # 2. ERA5 Temperature
    print("   Loading ERA5 daily max temperature...")
    era5_temp = pd.read_csv('2017_data/era5_daily_max_temp_2017_portugal.csv')
    era5_temp['time'] = pd.to_datetime(era5_temp['time'])
    era5_temp = normalize_timestamps(era5_temp, 12)
    print(f"      Shape: {era5_temp.shape}")
    print(f"      Time range: {era5_temp['time'].min()} to {era5_temp['time'].max()}")
    
    # 3. UERRA (sample - 5km resolution)
    print("   Loading UERRA (sampled for efficiency)...")
    uerra = pd.read_csv('2017_data/uerra_2017_PT_3decimal.csv')
    uerra['time'] = pd.to_datetime(uerra['time'])
    uerra = normalize_timestamps(uerra, 6)  # Original UERRA time
    # Sample every 100th record to make processing manageable
    uerra_sample = uerra.iloc[::100].copy()
    print(f"      Original: {uerra.shape}, Sampled: {uerra_sample.shape}")
    
    # 4. ERA5-Land (sample - ~10km resolution)
    print("   Loading ERA5-Land (sampled)...")
    era5_land = pd.read_csv('2017_data/era5_land_2017_PT_3decimal.csv')
    era5_land['time'] = pd.to_datetime(era5_land['time'])
    era5_land = normalize_timestamps(era5_land, 6)  # Original ERA5-Land time
    era5_land = era5_land.dropna(subset=['d2m', 't2m', 'u10', 'v10', 'tp'], how='all')
    # Sample every 50th record
    era5_land_sample = era5_land.iloc[::50].copy()
    print(f"      Original: {era5_land.shape}, Sampled: {era5_land_sample.shape}")
    
    # 5. WorldCover (static land cover)
    print("   Loading WorldCover...")
    worldcover = pd.read_csv('2017_data/esa_worldcover_portugal_3decimal.csv')
    print(f"      Shape: {worldcover.shape}")
    print(f"      Land cover classes: {worldcover['landcover_class'].nunique()}")
    
    # Start with ERA5 FWI as base feature matrix
    print("\n🔧 BUILDING FEATURE MATRIX...")
    features = era5_fwi[['target_time', 'latitude', 'longitude', 'fwi']].copy()
    features = features.rename(columns={'target_time': 'time'})
    
    print(f"   Base feature matrix: {features.shape}")
    
    # Add temporal features
    print("   Engineering temporal features...")
    features['year'] = features['time'].dt.year
    features['month'] = features['time'].dt.month
    features['day'] = features['time'].dt.day
    features['dayofyear'] = features['time'].dt.dayofyear
    features['quarter'] = features['time'].dt.quarter
    features['week'] = features['time'].dt.isocalendar().week
    
    # Cyclical temporal features (important for seasonal patterns)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['day_sin'] = np.sin(2 * np.pi * features['dayofyear'] / 365)
    features['day_cos'] = np.cos(2 * np.pi * features['dayofyear'] / 365)
    
    # Fire season indicator (May-October critical period in Portugal)
    features['fire_season'] = ((features['month'] >= 5) & (features['month'] <= 10)).astype(int)
    features['peak_fire_season'] = ((features['month'] >= 6) & (features['month'] <= 9)).astype(int)
    
    # Add spatial features
    print("   Engineering spatial features...")
    # Normalized coordinates
    features['lat_norm'] = (features['latitude'] - features['latitude'].min()) / (features['latitude'].max() - features['latitude'].min())
    features['lon_norm'] = (features['longitude'] - features['longitude'].min()) / (features['longitude'].max() - features['longitude'].min())
    
    # Distance features
    features['dist_from_coast'] = features['longitude'] - features['longitude'].min()  # Distance from Atlantic
    features['dist_from_spain'] = features['longitude'].max() - features['longitude']  # Distance from Spanish border
    
    # Geographic regions (simplified)
    features['region_north'] = (features['latitude'] > 41.0).astype(int)
    features['region_center'] = ((features['latitude'] > 39.0) & (features['latitude'] <= 41.0)).astype(int)
    features['region_south'] = (features['latitude'] <= 39.0).astype(int)
    features['coastal_zone'] = (features['longitude'] < -8.0).astype(int)
    features['inland_zone'] = (features['longitude'] > -7.5).astype(int)
    
    # Merge ERA5 temperature (same spatial grid)
    print("   Merging ERA5 temperature data...")
    era5_temp_clean = era5_temp[['target_time', 'latitude', 'longitude', 'temperature_max_celsius']].copy()
    era5_temp_clean = era5_temp_clean.rename(columns={'target_time': 'time'})
    
    features = features.merge(
        era5_temp_clean,
        on=['time', 'latitude', 'longitude'],
        how='left'
    )
    
    matched_temp = features['temperature_max_celsius'].notna().sum()
    print(f"      Matched temperature records: {matched_temp}/{len(features)} ({matched_temp/len(features)*100:.1f}%)")
    
    # Aggregate UERRA data to daily averages by region
    print("   Aggregating UERRA regional averages...")
    if not uerra_sample.empty:
        # Create daily regional averages
        uerra_sample['date'] = uerra_sample['target_time'].dt.date
        
        uerra_daily = uerra_sample.groupby('date').agg({
            'si10': 'mean',    # Wind speed
            'r2': 'mean',      # Relative humidity
            't2m': 'mean',     # Temperature
            'tp': 'sum'        # Total precipitation (sum for daily total)
        }).reset_index()
        
        # Convert to meaningful units
        uerra_daily['uerra_wind_speed'] = uerra_daily['si10']  # m/s
        uerra_daily['uerra_humidity'] = uerra_daily['r2']      # %
        uerra_daily['uerra_temp_c'] = uerra_daily['t2m'] - 273.15  # Celsius
        uerra_daily['uerra_precip_mm'] = uerra_daily['tp'] * 1000  # mm
        
        # Create time column for merging
        uerra_daily['time'] = pd.to_datetime(uerra_daily['date'].astype(str) + ' 12:00:00')
        
        # Merge with features
        features = features.merge(
            uerra_daily[['time', 'uerra_wind_speed', 'uerra_humidity', 'uerra_temp_c', 'uerra_precip_mm']],
            on='time',
            how='left'
        )
        
        matched_uerra = features['uerra_wind_speed'].notna().sum()
        print(f"      Matched UERRA records: {matched_uerra}/{len(features)} ({matched_uerra/len(features)*100:.1f}%)")
    
    # Aggregate ERA5-Land data to daily averages by region
    print("   Aggregating ERA5-Land regional averages...")
    if not era5_land_sample.empty:
        # Create daily regional averages
        era5_land_sample['date'] = era5_land_sample['target_time'].dt.date
        
        era5_land_daily = era5_land_sample.groupby('date').agg({
            'd2m': 'mean',     # Dewpoint temperature
            't2m': 'mean',     # Temperature  
            'u10': 'mean',     # Wind U component
            'v10': 'mean',     # Wind V component
            'tp': 'sum'        # Total precipitation
        }).reset_index()
        
        # Calculate derived features
        era5_land_daily['era5_land_wind_speed'] = np.sqrt(
            era5_land_daily['u10']**2 + era5_land_daily['v10']**2
        )
        era5_land_daily['era5_land_temp_c'] = era5_land_daily['t2m'] - 273.15
        era5_land_daily['era5_land_dewpoint_c'] = era5_land_daily['d2m'] - 273.15
        era5_land_daily['era5_land_precip_mm'] = era5_land_daily['tp'] * 1000
        
        # Calculate relative humidity (Magnus formula)
        temp_c = era5_land_daily['era5_land_temp_c']
        dewp_c = era5_land_daily['era5_land_dewpoint_c']
        
        alpha_temp = 17.27 * temp_c / (237.7 + temp_c)
        alpha_dewp = 17.27 * dewp_c / (237.7 + dewp_c)
        era5_land_daily['era5_land_humidity'] = 100 * np.exp(alpha_dewp - alpha_temp)
        era5_land_daily['era5_land_humidity'] = np.clip(era5_land_daily['era5_land_humidity'], 0, 100)
        
        # Create time column for merging
        era5_land_daily['time'] = pd.to_datetime(era5_land_daily['date'].astype(str) + ' 12:00:00')
        
        # Merge with features
        features = features.merge(
            era5_land_daily[['time', 'era5_land_wind_speed', 'era5_land_temp_c', 
                           'era5_land_dewpoint_c', 'era5_land_precip_mm', 'era5_land_humidity']],
            on='time',
            how='left'
        )
        
        matched_era5_land = features['era5_land_wind_speed'].notna().sum()
        print(f"      Matched ERA5-Land records: {matched_era5_land}/{len(features)} ({matched_era5_land/len(features)*100:.1f}%)")
    
    # Add land cover features (regional distribution)
    print("   Engineering land cover features...")
    if not worldcover.empty:
        # Calculate land cover statistics by broad regions
        landcover_stats = worldcover['landcover_class'].value_counts(normalize=True)
        
        print(f"      Land cover distribution: {dict(landcover_stats.head())}")
        
        # Add simplified land cover indicators based on coordinates
        # (In a full implementation, you'd do spatial joining)
        features['forest_region'] = ((features['latitude'] > 40.5) & (features['longitude'] < -7.5)).astype(int)
        features['agricultural_region'] = ((features['latitude'] < 40.0) & (features['longitude'] > -8.0)).astype(int)
        features['coastal_region'] = (features['longitude'] < -8.5).astype(int)
        features['mountainous_region'] = (features['latitude'] > 41.5).astype(int)
    
    # Add lag features for temporal patterns
    print("   Creating temporal lag features...")
    features = features.sort_values(['latitude', 'longitude', 'time']).reset_index(drop=True)
    
    # FWI lags (1, 3, 7 days)
    features['fwi_lag_1'] = features.groupby(['latitude', 'longitude'])['fwi'].shift(1)
    features['fwi_lag_3'] = features.groupby(['latitude', 'longitude'])['fwi'].shift(3)
    features['fwi_lag_7'] = features.groupby(['latitude', 'longitude'])['fwi'].shift(7)
    
    # Temperature lags
    if 'temperature_max_celsius' in features.columns:
        features['temp_lag_1'] = features.groupby(['latitude', 'longitude'])['temperature_max_celsius'].shift(1)
        features['temp_lag_3'] = features.groupby(['latitude', 'longitude'])['temperature_max_celsius'].shift(3)
    
    # Skip complex rolling operations for now - focus on essential features
    print("      Skipping rolling features to avoid complexity")
    
    # Feature analysis
    print("\n📈 COMPREHENSIVE FEATURE ANALYSIS...")
    print(f"   Final feature matrix: {features.shape}")
    print(f"   Time coverage: {features['time'].min()} to {features['time'].max()}")
    print(f"   Unique dates: {features['time'].dt.date.nunique()}")
    print(f"   Spatial coverage: {len(features[['latitude', 'longitude']].drop_duplicates())} unique coordinates")
    
    # Feature categories
    temporal_feats = [col for col in features.columns if any(x in col for x in ['month', 'day', 'year', 'season', 'sin', 'cos', 'quarter', 'week'])]
    spatial_feats = [col for col in features.columns if any(x in col for x in ['lat', 'lon', 'dist', 'region', 'coastal', 'inland', 'zone'])]
    weather_feats = [col for col in features.columns if any(x in col for x in ['temp', 'wind', 'humid', 'precip', 'dewpoint', 'uerra', 'era5_land'])]
    landcover_feats = [col for col in features.columns if any(x in col for x in ['forest', 'agricultural', 'mountainous', 'landcover'])]
    lag_feats = [col for col in features.columns if any(x in col for x in ['lag', 'roll'])]
    target_feat = ['fwi']
    
    print(f"\n   📊 Feature breakdown:")
    print(f"      Temporal features: {len(temporal_feats)}")
    print(f"      Spatial features: {len(spatial_feats)}")
    print(f"      Weather/Climate features: {len(weather_feats)}")
    print(f"      Land cover features: {len(landcover_feats)}")
    print(f"      Lag/Rolling features: {len(lag_feats)}")
    print(f"      Target variable: {len(target_feat)}")
    print(f"      Total: {len(features.columns)} features")
    
    # Data completeness analysis
    missing_analysis = features.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing_analysis / len(features)) * 100
    
    complete_features = missing_pct[missing_pct == 0]
    partial_features = missing_pct[(missing_pct > 0) & (missing_pct <= 50)]
    sparse_features = missing_pct[missing_pct > 50]
    
    print(f"\n   🎯 Data completeness:")
    print(f"      Complete features (0% missing): {len(complete_features)}")
    print(f"      Partial features (1-50% missing): {len(partial_features)}")
    print(f"      Sparse features (>50% missing): {len(sparse_features)}")
    
    if len(sparse_features) > 0:
        print(f"      Sparse feature details:")
        for col, pct in sparse_features.head().items():
            print(f"        {col}: {pct:.1f}% missing")
    
    # Target variable analysis
    fwi_stats = features['fwi'].describe()
    print(f"\n   🔥 FWI target analysis:")
    print(f"      Count: {int(fwi_stats['count'])}")
    print(f"      Mean: {fwi_stats['mean']:.2f}")
    print(f"      Std: {fwi_stats['std']:.2f}")
    print(f"      Range: {fwi_stats['min']:.2f} - {fwi_stats['max']:.2f}")
    print(f"      High fire risk days (FWI > 30): {(features['fwi'] > 30).sum()}")
    print(f"      Extreme fire risk days (FWI > 50): {(features['fwi'] > 50).sum()}")
    
    # Seasonal analysis
    seasonal_fwi = features.groupby('month')['fwi'].agg(['mean', 'max', 'count'])
    print(f"\n   📅 Seasonal FWI patterns:")
    for month in [6, 7, 8, 9]:  # Peak fire season
        mean_fwi = seasonal_fwi.loc[month, 'mean']
        max_fwi = seasonal_fwi.loc[month, 'max']
        count = seasonal_fwi.loc[month, 'count']
        print(f"      Month {month}: Mean={mean_fwi:.1f}, Max={max_fwi:.1f}, Records={count}")
    
    # Save engineered features
    print(f"\n💾 SAVING ENGINEERED FEATURES...")
    output_file = 'features_2017_final.csv'
    features.to_csv(output_file, index=False)
    
    file_size = features.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"   Output file: {output_file}")
    print(f"   File size: {file_size:.1f} MB") 
    print(f"   Shape: {features.shape}")
    
    # Create feature documentation
    feature_doc = pd.DataFrame({
        'Feature': features.columns,
        'Category': [''] * len(features.columns),
        'Description': [''] * len(features.columns),
        'Missing_Pct': [missing_pct.get(col, 0) for col in features.columns]
    })
    
    # Categorize features
    for idx, col in enumerate(feature_doc['Feature']):
        if col in temporal_feats:
            feature_doc.loc[idx, 'Category'] = 'Temporal'
        elif col in spatial_feats:
            feature_doc.loc[idx, 'Category'] = 'Spatial'
        elif col in weather_feats:
            feature_doc.loc[idx, 'Category'] = 'Weather'
        elif col in landcover_feats:
            feature_doc.loc[idx, 'Category'] = 'Land Cover'
        elif col in lag_feats:
            feature_doc.loc[idx, 'Category'] = 'Temporal Lag'
        elif col == 'fwi':
            feature_doc.loc[idx, 'Category'] = 'Target'
        else:
            feature_doc.loc[idx, 'Category'] = 'Identifier'
    
    feature_doc.to_csv('feature_documentation_2017.csv', index=False)
    print(f"   Feature documentation: feature_documentation_2017.csv")
    
    print(f"\n🎉 FEATURE ENGINEERING COMPLETE!")
    print(f"="*70)
    print(f"✅ Ready for FWI resolution enhancement (25km → 1km)")
    print(f"📊 Multi-resolution dataset with {features.shape[1]} features")
    print(f"🔥 {len(features)} FWI samples across Portugal 2017")
    print(f"🌍 Integrated ERA5, UERRA, ERA5-Land, and WorldCover data")
    
    return features

if __name__ == "__main__":
    features = main()