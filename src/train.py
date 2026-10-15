import os, time, json
import numpy as np
import pandas as pd
import joblib
import sklearn
import matplotlib
matplotlib.use("Agg")          # save charts without opening windows
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)
LABELS = ['positive', 'neutral', 'negative']

# ---------------------------------------------------------------
# 1. Load data and split
# ---------------------------------------------------------------
df = pd.read_csv('data/processed/clean_reviews.csv')
df = df.dropna(subset=['clean', 'sentiment'])
print("Total reviews:", len(df))

X_train, X_test, y_train, y_test, raw_train, raw_test, score_train, score_test = train_test_split(
    df['clean'], df['sentiment'], df['content'], df['score'],
    test_size=0.2, stratify=df['sentiment'], random_state=42)
print("Train:", len(X_train), "| Test:", len(X_test))

# ---------------------------------------------------------------
# 2. Compare models with 5-fold cross-validation
# ---------------------------------------------------------------
def make_pipe(clf):
    return Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000,
                                  sublinear_tf=True)),
        ('clf', clf)])

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'LinearSVC': LinearSVC(class_weight='balanced'),
    'ComplementNB': ComplementNB(),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for name, clf in models.items():
    start = time.time()
    scores = cross_validate(make_pipe(clf), X_train, y_train, cv=cv,
                            scoring=['accuracy', 'f1_macro'], n_jobs=-1)
    rows.append({
        'Model': name,
        'CV Accuracy': scores['test_accuracy'].mean(),
        'CV Macro F1': scores['test_f1_macro'].mean(),
        'Time (s)': time.time() - start,
    })
    print(f"{name:20s} acc={rows[-1]['CV Accuracy']:.4f}  macroF1={rows[-1]['CV Macro F1']:.4f}")

comparison = pd.DataFrame(rows).sort_values('CV Macro F1', ascending=False).round(4)
comparison.to_csv('reports/model_comparison.csv', index=False)
print("\n=== MODEL COMPARISON ===")
print(comparison.to_string(index=False))

best_name = comparison.iloc[0]['Model']
print("\nBest model before tuning:", best_name)

# ---------------------------------------------------------------
# 3. Hyperparameter tuning (GridSearchCV) on the best model
# ---------------------------------------------------------------
param_grids = {
    'LogisticRegression': {'clf__C': [0.5, 1, 2, 5, 10],
                           'tfidf__ngram_range': [(1, 1), (1, 2)]},
    'LinearSVC': {'clf__C': [0.1, 0.5, 1, 2],
                  'tfidf__ngram_range': [(1, 1), (1, 2)]},
    'ComplementNB': {'clf__alpha': [0.1, 0.3, 1.0],
                     'tfidf__ngram_range': [(1, 1), (1, 2)]},
}

grid = GridSearchCV(make_pipe(models[best_name]), param_grids[best_name],
                    cv=cv, scoring='f1_macro', n_jobs=-1)
grid.fit(X_train, y_train)
print("\nBest parameters:", grid.best_params_)
print("Best CV macro F1:", round(grid.best_score_, 4))

final_model = grid.best_estimator_

# ---------------------------------------------------------------
# 4. Final evaluation on the held-out test set
# ---------------------------------------------------------------
y_pred = final_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print("\n=== TEST SET RESULTS ===")
print("Accuracy :", round(acc, 4))
print("Macro F1 :", round(macro_f1, 4))
print(classification_report(y_test, y_pred, labels=LABELS))

# Confusion matrix (counts + row-normalised)
cm = confusion_matrix(y_test, y_pred, labels=LABELS)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS,
            yticklabels=LABELS, ax=axes[0])
axes[0].set_title("Confusion Matrix (counts)")
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=LABELS,
            yticklabels=LABELS, ax=axes[1])
axes[1].set_title("Confusion Matrix (row-normalised)")
for ax in axes:
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig('reports/figures/11_confusion_matrix.png', dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Error analysis
# ---------------------------------------------------------------
errors = pd.DataFrame({
    'review': raw_test.values, 'stars': score_test.values,
    'actual': y_test.values, 'predicted': y_pred
})
errors = errors[errors['actual'] != errors['predicted']]
errors.to_csv('reports/misclassified_reviews.csv', index=False)
print(f"\nMisclassified: {len(errors)} of {len(y_test)} ({len(errors)/len(y_test)*100:.1f}%)")
print("\nSample of misclassified reviews:")
for _, r in errors.sample(min(10, len(errors)), random_state=1).iterrows():
    print(f"- [{r['stars']} stars] actual={r['actual']}, predicted={r['predicted']}")
    print(f"  {str(r['review'])[:150]}")

# ---------------------------------------------------------------
# 6. Most important words (for linear models)
# ---------------------------------------------------------------
clf = final_model.named_steps['clf']
if hasattr(clf, 'coef_'):
    feats = np.array(final_model.named_steps['tfidf'].get_feature_names_out())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {'positive': 'green', 'neutral': 'gold', 'negative': 'red'}
    for ax, cls in zip(axes, LABELS):
        idx = list(clf.classes_).index(cls)
        top = np.argsort(clf.coef_[idx])[-15:]
        ax.barh(feats[top], clf.coef_[idx][top], color=colors[cls])
        ax.set_title(f"Top features: {cls}")
    plt.tight_layout()
    plt.savefig('reports/figures/12_top_features.png', dpi=150)
    plt.close()

# ---------------------------------------------------------------
# 7. Save the final model and metrics
# ---------------------------------------------------------------
joblib.dump(final_model, 'models/sentiment_pipeline.joblib')
metrics = {
    'best_model': best_name, 'best_params': {k: str(v) for k, v in grid.best_params_.items()},
    'test_accuracy': round(acc, 4), 'test_macro_f1': round(macro_f1, 4),
    'sklearn_version': sklearn.__version__, 'train_size': len(X_train), 'test_size': len(X_test)
}
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: models/sentiment_pipeline.joblib")
print("scikit-learn version (use this in requirements.txt):", sklearn.__version__)