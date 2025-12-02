"""Over/Underfitting analysis with learning curves."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ..data.processing import make_dataset, get_feature_columns


def plot_learning_curves(save_path=None):
    """
    Generate learning curves to analyze over/underfitting.
    
    Shows how train and validation accuracy change with more data.
    - If both are low: Underfitting (high bias)
    - If train high, val low: Overfitting (high variance)
    - If both converge high: Good fit
    """
    # Load data
    df = make_dataset()
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df['team_a_won'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Models to analyze
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), X_scaled),
        'XGBoost': (XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=42), X),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), X),
        'SVM': (SVC(kernel='rbf', random_state=42), X_scaled)
    }
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    print("=" * 70)
    print("OVER/UNDERFITTING ANALYSIS")
    print("=" * 70)
    print("\nGenerating learning curves...")
    
    for idx, (name, (model, X_data)) in enumerate(models.items()):
        print(f"  Processing {name}...")
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_data, y,
            train_sizes=train_sizes,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate means and stds
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        # Plot
        ax = axes[idx]
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='orange')
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training score')
        ax.plot(train_sizes_abs, val_mean, 'o-', color='orange', label='Validation score')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{name}')
        ax.legend(loc='lower right')
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, alpha=0.3)
        
        # Add gap annotation
        gap = train_mean[-1] - val_mean[-1]
        ax.annotate(f'Gap: {gap:.1%}', xy=(0.95, 0.05), xycoords='axes fraction', 
                   ha='right', fontsize=10, style='italic')
    
    plt.suptitle('Learning Curves - Over/Underfitting Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to {save_path}")
    
    plt.show()
    
    # Print analysis
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print("""
Learning Curve Interpretation:
─────────────────────────────────────────────────────────────────────
• UNDERFITTING (High Bias):
  - Both train and validation scores are LOW
  - Curves converge but at low accuracy
  - Solution: More complex model, more features

• OVERFITTING (High Variance):
  - Train score HIGH, validation score LOW
  - Large gap between curves
  - Solution: More data, regularization, simpler model

• GOOD FIT:
  - Both curves converge at HIGH accuracy
  - Small gap between train and validation
  - This is what we want!
─────────────────────────────────────────────────────────────────────
""")
    
    return fig


def analyze_overfitting():
    """Print detailed overfitting analysis."""
    df = make_dataset()
    feature_cols = get_feature_columns()
    X = df[feature_cols].values
    y = df['team_a_won'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("=" * 70)
    print("TRAIN vs VALIDATION ACCURACY (Overfitting Check)")
    print("=" * 70)
    print()
    
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), X_scaled),
        'XGBoost': (XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=42), X),
        'Random Forest': (RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42), X),
        'SVM': (SVC(kernel='rbf', random_state=42), X_scaled)
    }
    
    print(f"{'Model':<22} {'Train Acc':<12} {'Val Acc':<12} {'Gap':<12} {'Status'}")
    print("-" * 70)
    
    for name, (model, X_data) in models.items():
        # Fit on full data for train accuracy
        model.fit(X_data, y)
        train_acc = model.score(X_data, y)
        
        # Cross-val for validation accuracy
        val_acc = cross_val_score(model, X_data, y, cv=5).mean()
        
        gap = train_acc - val_acc
        
        if gap > 0.15:
            status = "⚠️  Overfitting"
        elif gap < 0.05 and val_acc < 0.65:
            status = "⚠️  Underfitting"
        else:
            status = "✅ Good fit"
        
        print(f"{name:<22} {train_acc:.1%}        {val_acc:.1%}        {gap:+.1%}        {status}")
    
    print("-" * 70)


if __name__ == "__main__":
    analyze_overfitting()
    print()
    plot_learning_curves('learning_curves.png')

