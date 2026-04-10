import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

print("=" * 50)
print("CREDIT CARD FRAUD DETECTION PROJECT")
print("=" * 50)

# 1. LOAD DATA
print("\n[1] Loading dataset...")
df = pd.read_csv('creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} out of {len(df)}")
print(f"Fraud percentage: {(df['Class'].sum()/len(df))*100:.2f}%")

# 2. VISUALIZE CLASS IMBALANCE
plt.figure(figsize=(8, 5))
sns.countplot(x=df['Class'])
plt.title('Fraud vs Non-Fraud Distribution')
plt.xlabel('Class (0=Non-Fraud, 1=Fraud)')
plt.ylabel('Count')
plt.savefig('class_distribution.png')
print("\n✓ Saved: class_distribution.png")

# 3. SPLIT DATA
print("\n[2] Splitting data...")
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 4. HANDLE IMBALANCE WITH SMOTE
print("\n[3] Applying SMOTE (handling imbalance)...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"After SMOTE - Training set: {X_train_balanced.shape[0]} samples")

# 5. TRAIN RANDOM FOREST
print("\n[4] Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)

# 6. PREDICT & EVALUATE
print("\n[5] Evaluating Random Forest...")
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_rf):.4f}")

# 7. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("✓ Saved: confusion_matrix.png")

# 8. ROC CURVE
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba_rf):.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.savefig('roc_curve.png')
print("✓ Saved: roc_curve.png")

# 9. FEATURE IMPORTANCE
print("\n[6] Feature Importance Analysis...")
importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Top 15 Most Important Features")
plt.bar(range(15), importances[indices][:15], align='center')
plt.xticks(range(15), [features[i] for i in indices[:15]], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✓ Saved: feature_importance.png")

# 10. ALSO TRY LOGISTIC REGRESSION FOR COMPARISON
print("\n[7] Training Logistic Regression (for comparison)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")

# 11. MODEL COMPARISON
print("\n" + "=" * 50)
print("MODEL COMPARISON SUMMARY")
print("=" * 50)
print(f"Random Forest ROC-AUC:      {roc_auc_score(y_test, y_pred_proba_rf):.4f}")
print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, y_pred_proba_lr):.4f}")
print("=" * 50)

print("\n✅ PROJECT COMPLETE! Check your folder for PNG files.")