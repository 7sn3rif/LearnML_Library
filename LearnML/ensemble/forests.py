import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import Arc
from supervised.trees import DecisionTree 


class RandomForestClassifier:
  def __init__(self, n_trees=20, min_samples_split=2, max_depth=10, n_features=None, criterion="gini"):
    self.n_trees = n_trees
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.n_features = n_features
    self.criterion = criterion
    self.trees = []


# helper function , to create a bootstrap sample of the dataset by randomly selecting samples with replacement. It takes the input features X and target labels y, and returns a new set of features and labels that can be used to train an individual decision tree in the random forest.
  def _bootstrap_sampling(self, X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True) # returns a new array of indices, where each index is randomly selected from the range of 0 to n_samples - 1. The size of this new array is also n_samples, and since replace=True, some indices may appear multiple times while others may not appear at all.
    return X[idxs], y[idxs]
  
    
def   fit(self, X, y):
  #create a forest with n trees
  for _ in range(self.n_trees):
    tree = DecisionTree(
        min_samples_split=self.min_samples_split,
        max_depth=self.max_depth,
        n_features=self.n_features,
        criterion=self.criterion
    )
    X_sample, y_sample = self._bootstrap_sampling(X, y)
    tree.fit(X_sample, y_sample)
    self.trees.append(tree)

    
          # ── Graphical representation of the Random Forest training pipeline ──────────
    fig, ax = plt.subplots(figsize=(9, 12))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    # ── Header ──────────────────────────────────────────────────────────────────
    header = mpatches.FancyBboxPatch((0.12, 0.918), 0.76, 0.062,
        boxstyle="round,pad=0.012", facecolor="#1E293B",
        edgecolor="#10B981", linewidth=1.8, zorder=2)
    ax.add_patch(header)
    
    ax.text(0.5, 0.949, "LEARNML  |  Random Forest Classifier Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3)
    
    
    # ── Steps ───────────────────────────────────────────────────────────────────
    steps = [
        {
            "y": 0.830,
            "title": "1. Initialize Forest",
            "sub": "n_trees = T  |  max_depth = d  |  criterion = gini",
            "color": "#38BDF8", "edge": "#0284C7",
        },
        {
            "y": 0.672,
            "title": "2. Bootstrap Sampling",
            "sub": "X_sample, y_sample = sample(X, y, replace=True)",
            "color": "#A855F7", "edge": "#7E22CE",
        },
        {
            "y": 0.514,
            "title": "3. Create & Train Decision Tree",
            "sub": "tree = DecisionTree(...)  →  tree.fit(X_sample, y_sample)",
            "color": "#EC4899", "edge": "#BE185D",
        },
        {
            "y": 0.356,
            "title": "4. Store Tree in Forest",
            "sub": "trees.append(tree)  |  one trained tree added",
            "color": "#F59E0B", "edge": "#B45309",
        },
        {
            "y": 0.198,
            "title": "5. Repeat for n_trees",
            "sub": "Bootstrap → Train → Store  |  repeat until T trees",
            "color": "#FB923C", "edge": "#C2410C",
        },
        {
            "y": 0.062,
            "title": "6. Majority Voting",
            "sub": "pred = argmax(count(tree_predictions))  →  final class",
            "color": "#10B981", "edge": "#059669",
        },
    ]
    
    card_w, card_h = 0.64, 0.090
    gap = 0.011
    
    
    # ── Cards ───────────────────────────────────────────────────────────────────
    for step in steps:
    
        cy = step["y"]
    
        x0 = 0.5 - card_w / 2
        y0 = cy - card_h / 2
    
        # Shadow
        shadow = mpatches.FancyBboxPatch(
            (x0 + 0.006, y0 - 0.006),
            card_w,
            card_h,
            boxstyle="round,pad=0.012",
            facecolor="#000000",
            edgecolor="none",
            alpha=0.35,
            zorder=2
        )
    
        ax.add_patch(shadow)
    
        # Card
        card = mpatches.FancyBboxPatch(
            (x0, y0),
            card_w,
            card_h,
            boxstyle="round,pad=0.012",
            facecolor="#1E293B",
            edgecolor=step["edge"],
            linewidth=2,
            zorder=3
        )
    
        ax.add_patch(card)
    
        # Left accent bar
        accent = mpatches.FancyBboxPatch(
            (x0, y0),
            0.013,
            card_h,
            boxstyle="round,pad=0.0",
            facecolor=step["color"],
            edgecolor="none",
            zorder=4
        )
    
        ax.add_patch(accent)
    
        # Title
        ax.text(
            0.5,
            cy + 0.020,
            step["title"],
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=step["color"],
            fontfamily="sans-serif",
            zorder=5
        )
    
        # Subtitle
        ax.text(
            0.5,
            cy - 0.022,
            step["sub"],
            ha="center",
            va="center",
            fontsize=9.5,
            color="#94A3B8",
            fontfamily="monospace",
            zorder=5
        )
    
    
    # ── Vertical arrows ─────────────────────────────────────────────────────────
    for i in range(len(steps) - 1):
    
        y_start = steps[i]["y"] - card_h / 2 - gap
        y_end = steps[i + 1]["y"] + card_h / 2 + gap
    
        ax.annotate(
            "",
            xy=(0.5, y_end),
            xytext=(0.5, y_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#475569",
                lw=1.8,
                mutation_scale=16
            ),
            zorder=3
        )
    
    
    # ── Forest construction loop: Step 5 → Step 2 ──────────────────────────────
    verts = [
        (0.818, steps[4]["y"]),
        (0.970, steps[3]["y"]),
        (0.970, steps[2]["y"]),
        (0.818, steps[1]["y"]),
    ]
    
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO
    ]
    
    path = Path(verts, codes)
    
    loop_patch = mpatches.PathPatch(
        path,
        facecolor="none",
        edgecolor="#F59E0B",
        linewidth=2,
        linestyle=(0, (5, 4)),
        zorder=3
    )
    
    ax.add_patch(loop_patch)
    
    
    # ── Arrowhead pointing back to Bootstrap Sampling ───────────────────────────
    ax.annotate(
        "",
        xy=(0.820, steps[1]["y"]),
        xytext=(0.845, steps[1]["y"] + 0.032),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#F59E0B",
            lw=1.5,
            mutation_scale=14
        ),
        zorder=4
    )
    
    
    # ── Forest loop label ───────────────────────────────────────────────────────
    ax.text(
        0.978,
        0.435,
        "Forest\nLoop",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#F59E0B",
        fontfamily="monospace",
        zorder=5
    )
    
    ax.text(
        0.978,
        0.385,
        "until n_trees\nreached",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#64748B",
        fontfamily="monospace",
        zorder=5
    )
    
    
    # ── Bootstrap annotation on Step 2 ──────────────────────────────────────────
    ax.text(
        0.845,
        steps[1]["y"] + 0.014,
        "RANDOM",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#0F172A",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#A855F7",
            edgecolor="none"
        ),
        zorder=6
    )
    
    
    # ── Tree annotation on Step 3 ───────────────────────────────────────────────
    ax.text(
        0.155,
        steps[2]["y"],
        "Tree 1\nTree 2\n...\nTree T",
        ha="center",
        va="center",
        fontsize=8,
        color="#EC4899",
        fontfamily="monospace",
        zorder=5
    )
    
    
    # ── Voting annotation on Step 6 ─────────────────────────────────────────────
    ax.text(
        0.155,
        steps[5]["y"],
        "0  1  1\n1  1  0\n0  1  1",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#10B981",
        fontfamily="monospace",
        zorder=5
    )
    
    ax.text(
        0.845,
        steps[5]["y"],
        "MAJORITY",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="#0F172A",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#10B981",
            edgecolor="none"
        ),
        zorder=6
    )
    
    
    # ── Final output arrow ──────────────────────────────────────────────────────
    ax.annotate(
        "",
        xy=(0.5, 0.008),
        xytext=(0.5, steps[5]["y"] - card_h / 2 - 0.01),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#10B981",
            lw=1.8,
            mutation_scale=15
        ),
        zorder=3
    )
    
    
    ax.text(
        0.5,
        -0.015,
        "FINAL CLASS PREDICTION",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#10B981",
        fontfamily="monospace",
        zorder=5
    )
    
    
    plt.tight_layout()
    
    plt.savefig(
        "random_forest_pipeline.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0F172A"
    )
    
    plt.show()
    























  def _voting(self,y):
   most_common=np.bincount(y).argmax()
   return most_common


  def predict(self,X):
    predictions = np.array([tree.predict(X) for tree in self.trees])
    predictions = np.transpose(predictions)
    final_predictions=[]
    for pred in predictions:
      final_predictions.append(self._voting(pred))
    return np.array(final_predictions)



