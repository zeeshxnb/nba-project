"""Playoff bracket predictor."""

import numpy as np
import pandas as pd
from ..models.logistic import LogisticModel
from ..data.processing import make_dataset, get_feature_columns, build_matchup_features
from ..data.ingest import load_team_stats


class PlayoffBracket:
    """Simulate NBA playoff bracket and predict champion."""
    
    def __init__(self, model=None):
        self.model = model
        self.feature_cols = get_feature_columns()
        self.team_stats = None
        self._trained = False
        
    def train(self):
        """Train model on all historical data."""
        df = make_dataset()
        X = df[self.feature_cols].values
        y = df['team_a_won'].values
        
        if self.model is None:
            self.model = LogisticModel()
        
        self.model.train(X, y, feature_names=self.feature_cols)
        self.team_stats = load_team_stats()
        self._trained = True
        
    def predict_series(self, team_a, team_b, season):
        """Predict winner of a single playoff series."""
        if not self._trained:
            self.train()
            
        features = self._build_single_matchup_features(team_a, team_b, season)
        
        if features is None:
            return {'error': f'Could not find stats for {team_a} or {team_b} in {season}'}
        
        X = np.array([features])
        prob_a = self.model.predict_proba(X)[0]
        
        winner = team_a if prob_a >= 0.5 else team_b
        confidence = prob_a if prob_a >= 0.5 else 1 - prob_a
        
        return {
            'team_a': team_a,
            'team_b': team_b,
            'winner': winner,
            'probability': confidence,
            'team_a_prob': prob_a
        }
    
    def predict_bracket(self, west_seeds, east_seeds, season):
        """Simulate full playoff bracket."""
        if not self._trained:
            self.train()
            
        results = {
            'west': {'round1': [], 'round2': [], 'conf_finals': None},
            'east': {'round1': [], 'round2': [], 'conf_finals': None},
            'finals': None,
            'champion': None
        }
        
        matchup_order = [(0, 7), (3, 4), (2, 5), (1, 6)]
        
        print("=" * 60)
        print(f"NBA PLAYOFF BRACKET PREDICTION - {season}")
        print("=" * 60)
        
        # First Round
        print("\nFIRST ROUND")
        print("-" * 60)
        
        west_r2 = []
        east_r2 = []
        
        for conf, seeds, r2_winners in [('WEST', west_seeds, west_r2), ('EAST', east_seeds, east_r2)]:
            print(f"\n{conf}ERN CONFERENCE:")
            for i, (high, low) in enumerate(matchup_order):
                result = self.predict_series(seeds[high], seeds[low], season)
                
                if 'error' in result:
                    print(f"  ERROR: {result['error']}")
                    continue
                    
                winner = result['winner']
                prob = result['probability']
                
                print(f"  ({high+1}) {seeds[high]} vs ({low+1}) {seeds[low]}: "
                      f"{winner} wins ({prob:.0%})")
                
                r2_winners.append(winner)
                
                if conf == 'WEST':
                    results['west']['round1'].append(result)
                else:
                    results['east']['round1'].append(result)
        
        # Second Round
        print("\nSECOND ROUND")
        print("-" * 60)
        
        west_cf = []
        east_cf = []
        
        r2_matchups = [(0, 1), (2, 3)]
        
        for conf, r2, cf_winners in [('WEST', west_r2, west_cf), ('EAST', east_r2, east_cf)]:
            print(f"\n{conf}ERN CONFERENCE:")
            for i, (a, b) in enumerate(r2_matchups):
                result = self.predict_series(r2[a], r2[b], season)
                winner = result['winner']
                prob = result['probability']
                
                print(f"  {r2[a]} vs {r2[b]}: {winner} wins ({prob:.0%})")
                
                cf_winners.append(winner)
                
                if conf == 'WEST':
                    results['west']['round2'].append(result)
                else:
                    results['east']['round2'].append(result)
        
        # Conference Finals
        print("\nCONFERENCE FINALS")
        print("-" * 60)
        
        west_result = self.predict_series(west_cf[0], west_cf[1], season)
        west_champ = west_result['winner']
        print(f"  WEST: {west_cf[0]} vs {west_cf[1]}: {west_champ} wins ({west_result['probability']:.0%})")
        results['west']['conf_finals'] = west_result
        
        east_result = self.predict_series(east_cf[0], east_cf[1], season)
        east_champ = east_result['winner']
        print(f"  EAST: {east_cf[0]} vs {east_cf[1]}: {east_champ} wins ({east_result['probability']:.0%})")
        results['east']['conf_finals'] = east_result
        
        # Finals
        print("\nNBA FINALS")
        print("-" * 60)
        
        finals_result = self.predict_series(west_champ, east_champ, season)
        champion = finals_result['winner']
        print(f"  {west_champ} vs {east_champ}: {champion} wins ({finals_result['probability']:.0%})")
        
        results['finals'] = finals_result
        results['champion'] = champion
        
        print("\n" + "=" * 60)
        print(f"PREDICTED CHAMPION: {champion}")
        print("=" * 60)
        
        return results
    
    def _build_single_matchup_features(self, team_a, team_b, season):
        """Build feature vector for a single matchup."""
        from ..data.processing import DIFF_FEATURES, INVERSE_FEATURES
        
        abbrev_map = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA', 'Charlotte Bobcats': 'CHA', 'Chicago Bulls': 'CHI',
            'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
            'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU',
            'Indiana Pacers': 'IND', 'Los Angeles Clippers': 'LAC', 'LA Clippers': 'LAC',
            'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 
            'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
            'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
            'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
            'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
            'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
        }
        
        a_stats = self.team_stats[
            (self.team_stats['TEAM_NAME'].map(abbrev_map) == team_a) & 
            (self.team_stats['SEASON'] == season)
        ]
        b_stats = self.team_stats[
            (self.team_stats['TEAM_NAME'].map(abbrev_map) == team_b) & 
            (self.team_stats['SEASON'] == season)
        ]
        
        if len(a_stats) == 0 or len(b_stats) == 0:
            return None
            
        a_stats = a_stats.iloc[0]
        b_stats = b_stats.iloc[0]
        
        features = []
        
        for col in DIFF_FEATURES:
            features.append(a_stats[col] - b_stats[col])
        
        for col in INVERSE_FEATURES:
            features.append(b_stats[col] - a_stats[col])
        
        home_court = 1 if a_stats['W_PCT'] >= b_stats['W_PCT'] else 0
        features.append(home_court)
        
        return features


def predict_series(team_a, team_b, season='2022-23'):
    """Convenience function to predict a single series."""
    bracket = PlayoffBracket()
    return bracket.predict_series(team_a, team_b, season)


def predict_bracket(west_seeds, east_seeds, season='2022-23'):
    """Convenience function to predict full bracket."""
    bracket = PlayoffBracket()
    return bracket.predict_bracket(west_seeds, east_seeds, season)


if __name__ == "__main__":
    west_2023 = ['DEN', 'MEM', 'SAC', 'PHX', 'LAC', 'GSW', 'LAL', 'MIN']
    east_2023 = ['MIL', 'BOS', 'PHI', 'CLE', 'NYK', 'BKN', 'MIA', 'ATL']
    
    bracket = PlayoffBracket()
    results = bracket.predict_bracket(west_2023, east_2023, '2022-23')
