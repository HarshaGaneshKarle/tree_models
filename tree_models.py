import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("heart.csv")
print("🔹 Dataset Loaded:\n", df.head())

# Split data
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# Visualize Tree
plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.savefig("decision_tree.png")
plt.close()

# Evaluate Decision Tree
y_pred_tree = tree.predict(X_test)
print("\n🔹 Decision Tree Report:")
print(classification_report(y_test, y_pred_tree))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf.predict(X_test)
print("\n🔹 Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# Cross-validation
cv_score_tree = cross_val_score(tree, X, y, cv=5).mean()
cv_score_rf = cross_val_score(rf, X, y, cv=5).mean()

print(f"\nDecision Tree CV Accuracy: {cv_score_tree:.2f}")
print(f"Random Forest CV Accuracy: {cv_score_rf:.2f}")

# Feature importance (Random Forest)
feat_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_importances.plot(kind="bar", title="Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
