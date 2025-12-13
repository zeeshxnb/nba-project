"""
NBA Playoff Bracket Predictor
By Atif Usmani and Zeeshan Babul

Results (3-Year Test: 2020-21, 2021-22, 2022-23):
-------------------------------------------------
Best Model: Logistic Regression
Test Accuracy: 73% (30/41 series correct)
CV Accuracy: 74-75%

Season Breakdown:
  2020-21: 75% (9/12)
  2021-22: 87% (13/15)
  2022-23: 57% (8/14)
"""

import argparse

TEST_SEASONS = ['2020-21', '2021-22', '2022-23']


def run_evaluation():
    """Evaluate models on 3-year holdout test set."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION (3-YEAR TEST)")
    print("=" * 70)
    from src.evaluate import run_full_evaluation
    run_full_evaluation(test_seasons=TEST_SEASONS)


def run_tuning():
    """Run hyperparameter tuning with GridSearchCV."""
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    from src.evaluate import tune_all_models
    tune_all_models()


def run_overfitting_analysis():
    """Analyze train vs validation gap."""
    print("\n" + "=" * 70)
    print("OVERFITTING ANALYSIS")
    print("=" * 70)
    from src.evaluate import analyze_overfitting
    analyze_overfitting()


def run_monte_carlo(n_simulations=500):
    """Run Monte Carlo bracket simulation."""
    print("\n" + "=" * 70)
    print("MONTE CARLO SIMULATION")
    print("=" * 70)
    from src.evaluate.advanced import run_monte_carlo_bracket
    return run_monte_carlo_bracket(n_simulations=n_simulations)


def run_bracket_prediction(season='2022-23'):
    """Predict playoff bracket for a specific season."""
    print("\n" + "=" * 70)
    print(f"BRACKET PREDICTION ({season})")
    print("=" * 70)
    
    from src.predict import PlayoffBracket
    from src.models.ensemble import PlayoffPredictor
    
    seeds = {
        '2022-23': {
            'west': ['DEN', 'MEM', 'SAC', 'PHX', 'LAC', 'GSW', 'LAL', 'MIN'],
            'east': ['MIL', 'BOS', 'PHI', 'CLE', 'NYK', 'BKN', 'MIA', 'ATL']
        },
        '2021-22': {
            'west': ['PHX', 'MEM', 'GSW', 'DAL', 'UTA', 'DEN', 'MIN', 'NOP'],
            'east': ['MIA', 'BOS', 'MIL', 'PHI', 'TOR', 'CHI', 'BKN', 'ATL']
        },
        '2020-21': {
            'west': ['UTA', 'PHX', 'DEN', 'LAC', 'DAL', 'POR', 'LAL', 'MEM'],
            'east': ['PHI', 'BKN', 'MIL', 'NYK', 'ATL', 'MIA', 'BOS', 'WAS']
        }
    }
    
    if season not in seeds:
        print(f"Season {season} not available. Using 2022-23.")
        season = '2022-23'
    
    class EnsembleWrapper:
        def __init__(self):
            self.ensemble = PlayoffPredictor()
            self.is_trained = False
            self.feature_names = None
        def train(self, X, y, feature_names=None):
            self.feature_names = feature_names
            self.ensemble.train(X, y)
            self.is_trained = True
        def predict(self, X):
            return self.ensemble.predict(X)
        def predict_proba(self, X):
            return self.ensemble.predict_proba(X)
    
    bracket = PlayoffBracket(model=EnsembleWrapper())
    return bracket.predict_bracket(seeds[season]['west'], seeds[season]['east'], season)


def run_full_pipeline():
    """Run complete analysis pipeline."""
    print("=" * 70)
    print("NBA PLAYOFF BRACKET PREDICTOR")
    print("By Atif Usmani and Zeeshan Babul")
    print("=" * 70)
    print(f"\nTest Seasons: {', '.join(TEST_SEASONS)}")
    print("Models: Logistic Regression, XGBoost, Random Forest, SVM\n")
    
    run_evaluation()
    run_overfitting_analysis()
    run_bracket_prediction('2022-23')
    run_monte_carlo(n_simulations=500)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("""
Best Performing Model: Logistic Regression
  - CV Accuracy: 74.0%
  - 3-Year Test Accuracy: 73.2% (30/41 correct)

Season-by-Season Performance:
  - 2020-21: 75% (9/12 series)
  - 2021-22: 87% (13/15 series)  <- Best
  - 2022-23: 57% (8/14 series)   <- Most upsets

Key Insight: Simpler models generalize better than complex ones.
""")


def main():
    parser = argparse.ArgumentParser(description='NBA Playoff Bracket Predictor')
    parser.add_argument('--evaluate', action='store_true', help='Model evaluation (3-year test)')
    parser.add_argument('--tune', action='store_true', help='Hyperparameter tuning')
    parser.add_argument('--overfit', action='store_true', help='Overfitting analysis')
    parser.add_argument('--monte-carlo', action='store_true', help='Monte Carlo simulation')
    parser.add_argument('--predict', type=str, metavar='SEASON', help='Bracket prediction')
    
    args = parser.parse_args()
    
    if args.evaluate:
        run_evaluation()
    elif args.tune:
        run_tuning()
    elif args.overfit:
        run_overfitting_analysis()
    elif args.monte_carlo:
        run_monte_carlo(n_simulations=1000)
    elif args.predict:
        run_bracket_prediction(args.predict)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
