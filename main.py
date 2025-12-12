import argparse


def run_evaluation():
    """Run model comparison and evaluation."""
    print("\n" + "=" * 70)
    print("STEP 1: MODEL EVALUATION")
    print("=" * 70)
    
    from src.evaluate import run_full_evaluation
    run_full_evaluation()


def run_tuning():
    """Run hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("STEP 2: HYPERPARAMETER TUNING")
    print("=" * 70)
    
    from src.evaluate import tune_all_models
    tune_all_models()


def run_overfitting_analysis():
    """Run overfitting analysis."""
    print("\n" + "=" * 70)
    print("STEP 3: OVERFITTING ANALYSIS")
    print("=" * 70)
    
    from src.evaluate import analyze_overfitting
    analyze_overfitting()


def run_advanced_analysis():
    """Run feature selection and Monte Carlo analysis."""
    from src.evaluate import run_all_advanced
    run_all_advanced()


def run_bracket_prediction(season='2022-23'):
    """Predict playoff bracket for a given season."""
    print("\n" + "=" * 70)
    print(f"STEP 4: BRACKET PREDICTION ({season})")
    print("=" * 70)
    
    from src.predict import PlayoffBracket
    from src.models.ensemble import PlayoffPredictor
    
    playoff_seeds = {
        '2022-23': {
            'west': ['DEN', 'MEM', 'SAC', 'PHX', 'LAC', 'GSW', 'LAL', 'MIN'],
            'east': ['MIL', 'BOS', 'PHI', 'CLE', 'NYK', 'BKN', 'MIA', 'ATL']
        },
        '2021-22': {
            'west': ['PHX', 'MEM', 'GSW', 'DAL', 'UTA', 'DEN', 'MIN', 'NOP'],
            'east': ['MIA', 'BOS', 'MIL', 'PHI', 'TOR', 'CHI', 'BKN', 'ATL']
        }
    }
    
    if season not in playoff_seeds:
        print(f"Season {season} not available. Using 2022-23.")
        season = '2022-23'
    
    seeds = playoff_seeds[season]
    
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
    results = bracket.predict_bracket(seeds['west'], seeds['east'], season)
    
    return results


def run_full_pipeline():
    """Run the complete pipeline."""
    print("=" * 70)
    print("NBA PLAYOFF BRACKET PREDICTOR")
    print("=" * 70)
    print("\nPredicts playoff series winners using machine learning.")
    print("Models: Logistic Regression, XGBoost, Random Forest, SVM, Ensemble\n")
    
    run_evaluation()
    run_overfitting_analysis()
    run_bracket_prediction('2022-23')
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("- Per-series accuracy: ~75% (cross-validated)")
    print("- Champion prediction: ~35-40% (5-6x better than random)")
    print("- Best model: Ensemble (Logistic + XGBoost)")
    print("\nFor tuning: python main.py --tune")
    print("For learning curves: python -m src.evaluate.overfitting\n")


def main():
    parser = argparse.ArgumentParser(description='NBA Playoff Bracket Predictor')
    parser.add_argument('--evaluate', action='store_true', help='Run model evaluation')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--overfit', action='store_true', help='Run overfitting analysis')
    parser.add_argument('--advanced', action='store_true', help='Run advanced analysis')
    parser.add_argument('--predict', type=str, metavar='SEASON', help='Predict bracket for season')
    
    args = parser.parse_args()
    
    if args.evaluate:
        run_evaluation()
    elif args.tune:
        run_tuning()
    elif args.overfit:
        run_overfitting_analysis()
    elif args.advanced:
        run_advanced_analysis()
    elif args.predict:
        run_bracket_prediction(args.predict)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
