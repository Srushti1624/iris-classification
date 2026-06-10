import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
 
# ── 2. LOAD DATASET ───────────────────────────────────────────────────────────
print("=" * 60)
print("  IRIS FLOWER CLASSIFICATION")
print("=" * 60)
 
iris = load_iris()
 
# Build a clean DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"]      = iris.target
df["species_name"] = pd.Categorical.from_codes(iris.target, iris.target_names)
 
print("\n[1] Dataset loaded successfully.")
print(f"    Shape : {df.shape}  (rows, columns)")
print(f"    Classes: {list(iris.target_names)}")
print("\nFirst 5 rows:")
print(df.head())
 
# ── 3. EXPLORE DATA ───────────────────────────────────────────────────────────
print("\n[2] Basic statistics:")
print(df.describe().round(2))
 
print("\n    Missing values:", df.isnull().sum().sum(), "(none expected)")
print("    Class distribution:\n", df["species_name"].value_counts())
 
# --- 3a. Pairplot -------------------------------------------------------
print("\n[3] Generating pairplot...")
sns.pairplot(
    df,
    hue="species_name",
    vars=iris.feature_names,
    diag_kind="hist",
    palette="Set2",
)
plt.suptitle("Iris Dataset — Pairplot (all feature pairs)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig("plot_1_pairplot.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved → plot_1_pairplot.png")
 
# --- 3b. Boxplots -------------------------------------------------------
print("[4] Generating box plots...")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
colors = ["#66c2a5", "#fc8d62", "#8da0cb"]
 
for i, col in enumerate(iris.feature_names):
    data_by_species = [
        df.loc[df["species"] == j, col].values for j in range(3)
    ]
    bp = axes[i].boxplot(data_by_species, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[i].set_xticklabels(iris.target_names, rotation=15)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_ylabel("cm")
 
plt.suptitle("Feature Distributions by Species", fontsize=13)
plt.tight_layout()
plt.savefig("plot_2_boxplots.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved → plot_2_boxplots.png")
 
# ── 4. SPLIT DATA ─────────────────────────────────────────────────────────────
print("\n[5] Splitting data into train / test sets...")
 
X = df[iris.feature_names]   # Features: 4 numeric columns
y = df["species"]             # Target: 0, 1, or 2
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,     # 80% train, 20% test
    random_state=42,   # reproducible split
    stratify=y,        # keep class balance in both sets
)
 
print(f"    Training samples : {len(X_train)}")
print(f"    Test samples     : {len(X_test)}")
 
# ── 5. SCALE FEATURES ────────────────────────────────────────────────────────
# Scaling is required for Logistic Regression & KNN (distance-based).
# Decision Tree does NOT need scaling, but we scale for uniformity.
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled  = scaler.transform(X_test)        # transform only on test
 
print("\n[6] Features scaled (mean=0, std=1) using StandardScaler.")
 
# ── 6. TRAIN MODELS ──────────────────────────────────────────────────────────
print("\n[7] Training classifiers...")
 
models = {
    "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=4, random_state=42),
}
 
# Train all three models on scaled data
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    print(f"    ✓ {name} trained.")
 
# ── 7. EVALUATE MODELS ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODEL EVALUATION")
print("=" * 60)
 
results = {}   # Store accuracy for comparison chart
 
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    acc    = accuracy_score(y_test, y_pred)
    results[name] = acc
 
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"  Accuracy : {acc:.2%}")
    print(f"\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")
 
# ── 8. CONFUSION MATRICES ────────────────────────────────────────────────────
print("[8] Generating confusion matrices...")
 
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
 
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test_scaled)
    cm     = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        linewidths=0.5,
    )
    ax.set_title(f"{name}\nAccuracy: {results[name]:.2%}", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
 
plt.suptitle("Confusion Matrices — All Models", fontsize=13)
plt.tight_layout()
plt.savefig("plot_3_confusion_matrices.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved → plot_3_confusion_matrices.png")
 
# ── 9. ACCURACY COMPARISON BAR CHART ────────────────────────────────────────
print("[9] Generating accuracy comparison chart...")
 
plt.figure(figsize=(7, 4))
bars = plt.bar(
    results.keys(),
    [v * 100 for v in results.values()],
    color=["#66c2a5", "#fc8d62", "#8da0cb"],
    edgecolor="white",
    width=0.5,
)
for bar, val in zip(bars, results.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 4,
        f"{val:.1%}",
        ha="center", va="top", color="white", fontweight="bold", fontsize=11,
    )
plt.ylim(80, 102)
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("plot_4_accuracy_comparison.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved → plot_4_accuracy_comparison.png")
 
# ── 10. VISUALIZE DECISION TREE ──────────────────────────────────────────────
print("[10] Visualizing Decision Tree structure...")
 
plt.figure(figsize=(14, 5))
plot_tree(
    models["Decision Tree"],
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True, fontsize=9,
)
plt.title("Decision Tree — Max Depth 4", fontsize=13)
plt.tight_layout()
plt.savefig("plot_5_decision_tree.png", dpi=120, bbox_inches="tight")
plt.show()
print("    Saved → plot_5_decision_tree.png")
 
# ── 11. PREDICT A NEW FLOWER ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREDICTING A NEW FLOWER")
print("=" * 60)
 
# Format: [sepal length, sepal width, petal length, petal width]  (all in cm)
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
new_flower_scaled = scaler.transform(new_flower)
 
print(f"\n  Input measurements:")
print(f"    Sepal length : {new_flower[0][0]} cm")
print(f"    Sepal width  : {new_flower[0][1]} cm")
print(f"    Petal length : {new_flower[0][2]} cm")
print(f"    Petal width  : {new_flower[0][3]} cm")
 
best_model = models["Logistic Regression"]
prediction = best_model.predict(new_flower_scaled)
probability = best_model.predict_proba(new_flower_scaled)
 
print(f"\n  Predicted species : {iris.target_names[prediction[0]].upper()}")
print(f"  Confidence        : {probability.max():.1%}")
print(f"\n  Probabilities:")
for name, prob in zip(iris.target_names, probability[0]):
    bar = "█" * int(prob * 30)
    print(f"    {name:<12} {prob:.1%}  {bar}")
 
# ── 12. SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
best_name = max(results, key=results.get)
print(f"\n  {'Model':<25} {'Accuracy':>10}")
print(f"  {'─'*35}")
for name, acc in results.items():
    marker = " ← best" if name == best_name else ""
    print(f"  {name:<25} {acc:>9.2%}{marker}")
print(f"\n  Plots saved: plot_1 through plot_5")
print(f"\n  Done!\n")
