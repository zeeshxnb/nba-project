"""Train/test split utilities."""

import pandas as pd
import numpy as np
from ..data.processing import make_dataset, get_feature_columns

# Default: test on last 3 seasons
DEFAULT_TEST_SEASONS = ['2020-21', '2021-22', '2022-23']


def get_train_test_split(test_seasons=None):
    """Split data by season for time-based evaluation."""
    if test_seasons is None:
        test_seasons = DEFAULT_TEST_SEASONS
    
    if isinstance(test_seasons, str):
        test_seasons = [test_seasons]
    
    df = make_dataset()
    feature_cols = get_feature_columns()
    
    train_df = df[~df['season'].isin(test_seasons)]
    test_df = df[df['season'].isin(test_seasons)]
    
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_train = train_df['team_a_won'].values
    y_test = test_df['team_a_won'].values
    
    return X_train, X_test, y_train, y_test, feature_cols, df, test_df


def get_data_info(test_seasons=None):
    """Print train/test split information."""
    if test_seasons is None:
        test_seasons = DEFAULT_TEST_SEASONS
    
    if isinstance(test_seasons, str):
        test_seasons = [test_seasons]
    
    df = make_dataset()
    
    train_df = df[~df['season'].isin(test_seasons)]
    test_df = df[df['season'].isin(test_seasons)]
    
    print("=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"\nTraining Data:")
    print(f"  Seasons: {train_df['season'].min()} to {train_df['season'].max()}")
    print(f"  Samples: {len(train_df)} playoff series")
    print(f"  Target balance: {train_df['team_a_won'].mean():.1%} team_a wins")
    
    print(f"\nTest Data (Held Out):")
    print(f"  Seasons: {', '.join(test_seasons)} ({len(test_seasons)} years)")
    print(f"  Samples: {len(test_df)} playoff series")
    print(f"  Target balance: {test_df['team_a_won'].mean():.1%} team_a wins")
    
    print("\nTest Set Breakdown by Season:")
    for season in test_seasons:
        season_df = test_df[test_df['season'] == season]
        print(f"  {season}: {len(season_df)} series")
    
    print("=" * 60)
    
    return train_df, test_df
