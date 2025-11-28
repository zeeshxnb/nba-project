"""
Step 2.1: Calculate stat differentials
Step 2.2: Add seeding & home court features
"""
import pandas as pd
from .data_loader import load_team_stats, load_series_history

# Stats to compute differentials for (higher = better for team)
DIFF_FEATURES = [
    'NET_RATING', 'OFF_RATING', 'W_PCT', 'TS_PCT', 'EFG_PCT', 
    'OREB_PCT', 'FTA_RATE', 'OPP_TOV_PCT'
]
# Stats where lower is better (flip sign)
INVERSE_FEATURES = ['DEF_RATING', 'TM_TOV_PCT', 'OPP_EFG_PCT', 'OPP_OREB_PCT', 'OPP_FTA_RATE']


def build_matchup_features(series_df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for each historical series matchup.
    Features are differentials: team_a_stat - team_b_stat
    """
    # Map team abbreviation to full stats for each season
    stats_by_abbrev = _create_abbrev_mapping(team_stats)
    
    features = []
    for _, row in series_df.iterrows():
        season = row['season']
        team_a, team_b = row['team_a'], row['team_b']
        
        # Get team stats for this season
        a_stats = stats_by_abbrev.get((team_a, season))
        b_stats = stats_by_abbrev.get((team_b, season))
        
        if a_stats is None or b_stats is None:
            continue
        
        # Calculate differentials
        feat = {'season': season, 'team_a': team_a, 'team_b': team_b}
        
        for col in DIFF_FEATURES:
            feat[f'{col.lower()}_diff'] = a_stats[col] - b_stats[col]
        
        for col in INVERSE_FEATURES:
            # Flip sign: lower DEF_RATING is better, so we want (b - a)
            feat[f'{col.lower()}_diff'] = b_stats[col] - a_stats[col]
        
        # Home court: team with better record has it
        feat['home_court'] = 1 if a_stats['W_PCT'] >= b_stats['W_PCT'] else 0
        
        # Target: did team_a win?
        feat['team_a_won'] = 1 if row['winner'] == team_a else 0
        
        features.append(feat)
    
    return pd.DataFrame(features)


def _create_abbrev_mapping(team_stats: pd.DataFrame) -> dict:
    """Create (abbreviation, season) -> stats mapping."""
    # Team name to abbreviation mapping
    abbrev_map = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Charlotte Bobcats': 'CHA', 'Chicago Bulls': 'CHI',
        'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
        'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU',
        'Indiana Pacers': 'IND', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 
        'New Orleans Hornets': 'NOH', 'New Orleans/Oklahoma City Hornets': 'NOK',
        'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS', 'Washington Bullets': 'WAS',
        'Seattle SuperSonics': 'SEA', 'Vancouver Grizzlies': 'VAN', 'New Jersey Nets': 'NJN',
    }
    
    mapping = {}
    for _, row in team_stats.iterrows():
        abbrev = abbrev_map.get(row['TEAM_NAME'])
        if abbrev:
            mapping[(abbrev, row['SEASON'])] = row
    
    return mapping


def get_feature_columns() -> list:
    """Return list of feature column names (for model training)."""
    cols = [f'{c.lower()}_diff' for c in DIFF_FEATURES + INVERSE_FEATURES]
    cols.append('home_court')
    return cols


if __name__ == "__main__":
    # Test
    team_stats = load_team_stats()
    series = load_series_history()
    
    df = build_matchup_features(series, team_stats)
    print(f"Built {len(df)} matchup samples")
    print(f"Features: {get_feature_columns()}")
    print(f"\nSample:\n{df.head()}")

