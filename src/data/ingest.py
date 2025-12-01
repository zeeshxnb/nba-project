import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Define columns to keep
ADVANCED_COLS = ['TEAM_ID', 'TEAM_NAME', 'SEASON', 'NET_RATING', 'OFF_RATING', 'DEF_RATING', 'TS_PCT', 'PACE']
TRADITIONAL_COLS = ['TEAM_ID', 'SEASON', 'W', 'L', 'W_PCT']
FOUR_FACTORS_COLS = ['TEAM_ID', 'SEASON', 'EFG_PCT', 'FTA_RATE', 'TM_TOV_PCT', 'OREB_PCT', 
                     'OPP_EFG_PCT', 'OPP_FTA_RATE', 'OPP_TOV_PCT', 'OPP_OREB_PCT']

def load_team_stats() -> pd.DataFrame:
    """Load and merge all team stats into one DataFrame."""
    advanced = pd.read_csv(DATA_DIR / "team/team_stats_advanced_rs.csv", usecols=ADVANCED_COLS)
    traditional = pd.read_csv(DATA_DIR / "team/team_stats_traditional_rs.csv", usecols=TRADITIONAL_COLS)
    four_factors = pd.read_csv(DATA_DIR / "team/team_stats_four_factors_rs.csv", usecols=FOUR_FACTORS_COLS)
    
    # Merge on TEAM_ID and SEASON
    team_stats = advanced.merge(traditional, on=['TEAM_ID', 'SEASON'])
    team_stats = team_stats.merge(four_factors, on=['TEAM_ID', 'SEASON'])
    
    return team_stats

def load_playoff_history() -> pd.DataFrame:
    """Load historical playoff advanced stats."""
    cols = ['TEAM_ID', 'TEAM_NAME', 'SEASON', 'GP', 'W', 'L', 'NET_RATING']
    return pd.read_csv(DATA_DIR / "team/team_stats_advanced_po.csv", usecols=cols)

def load_series_history() -> pd.DataFrame:
    """Load historical playoff series matchups."""
    series_path = DATA_DIR / "playoff_series.csv"
    if not series_path.exists():
        raise FileNotFoundError(f"Missing {series_path}")
    return pd.read_csv(series_path)

