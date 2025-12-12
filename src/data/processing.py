"""Data processing and feature engineering."""

import pandas as pd
from .ingest import load_team_stats, load_series_history

# Features where higher is better
DIFF_FEATURES = [
    'NET_RATING', 'OFF_RATING', 'W_PCT', 'TS_PCT', 'EFG_PCT', 
    'OREB_PCT', 'FTA_RATE', 'OPP_TOV_PCT'
]

# Features where lower is better (sign flipped)
INVERSE_FEATURES = ['DEF_RATING', 'TM_TOV_PCT', 'OPP_EFG_PCT', 'OPP_OREB_PCT', 'OPP_FTA_RATE']


def build_matchup_features(series_df, team_stats):
    """Build feature matrix with differentials for each matchup."""
    stats_by_abbrev = _create_abbrev_mapping(team_stats)
    
    features = []
    for _, row in series_df.iterrows():
        season = row['season']
        team_a, team_b = row['team_a'], row['team_b']
        
        a_stats = stats_by_abbrev.get((team_a, season))
        b_stats = stats_by_abbrev.get((team_b, season))
        
        if a_stats is None or b_stats is None:
            continue
        
        feat = {'season': season, 'team_a': team_a, 'team_b': team_b}
        
        for col in DIFF_FEATURES:
            feat[f'{col.lower()}_diff'] = a_stats[col] - b_stats[col]
        
        for col in INVERSE_FEATURES:
            feat[f'{col.lower()}_diff'] = b_stats[col] - a_stats[col]
        
        feat['home_court'] = 1 if a_stats['W_PCT'] >= b_stats['W_PCT'] else 0
        
        if 'winner' in row:
            feat['team_a_won'] = 1 if row['winner'] == team_a else 0
        
        features.append(feat)
    
    return pd.DataFrame(features)


def _create_abbrev_mapping(team_stats):
    """Create (abbreviation, season) -> stats mapping."""
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


def get_feature_columns():
    """Return list of feature column names."""
    cols = [f'{c.lower()}_diff' for c in DIFF_FEATURES + INVERSE_FEATURES]
    cols.append('home_court')
    return cols


def make_dataset():
    """Load data and build feature matrix."""
    team_stats = load_team_stats()
    series_history = load_series_history()
    return build_matchup_features(series_history, team_stats)
