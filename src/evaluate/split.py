"""Train/Test split utilities for time-series playoff data."""

import pandas as pd
import numpy as np
from ..data.processing import make_dataset, get_feature_columns


def get_train_test_split(test_season='2022-23'):
    """
    Split data into train and test sets based on season.
    
    Args:
        test_season: Season to hold out for testing (default: 2023-24)
    
    Returns:
        X_train, X_test, y_train, y_test, feature_cols, df
    """
    df = make_dataset()
    feature_cols = get_feature_columns()
    
    # Split by season
    train_df = df[df['season'] != test_season]
    test_df = df[df['season'] == test_season]
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df['team_a_won'].values
    y_test = test_df['team_a_won'].values
    
    return X_train, X_test, y_train, y_test, feature_cols, df


def get_data_info(test_season='2022-23'):
    """Print information about the train/test split."""
    df = make_dataset()
    
    train_df = df[df['season'] != test_season]
    test_df = df[df['season'] == test_season]
    
    print("=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"\nTraining Data:")
    print(f"  Seasons: {train_df['season'].min()} to {train_df['season'].max()}")
    print(f"  Samples: {len(train_df)} playoff series")
    print(f"  Target balance: {train_df['team_a_won'].mean():.1%} team_a wins")
    
    print(f"\nTest Data (Held Out):")
    print(f"  Season: {test_season}")
    print(f"  Samples: {len(test_df)} playoff series")
    print(f"  Target balance: {test_df['team_a_won'].mean():.1%} team_a wins")
    
    print("\nTest Set Matchups:")
    for _, row in test_df.iterrows():
        winner = row['team_a'] if row['team_a_won'] == 1 else row['team_b']
        print(f"  {row['team_a']} vs {row['team_b']} → {winner}")
    
    print("=" * 60)
    
    return train_df, test_df

