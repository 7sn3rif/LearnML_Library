import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.path as Path
from collections import Counter


class KNN:
 def __init__(self, k=3, distance="euclidean"):
      self.k = k
      self.distance = distance
      # Define distance functions
      self.euclidean_distance = lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2))
      self.manhattan_distance = lambda x1, x2: np.sum(np.abs(x1 - x2))
      self.minkowski_distance = lambda x1, x2, p=3: np.sum(np.abs(x1 - x2) ** p) ** (1/p)

 #this function is used to return the required distance metric
 def _compute_distance(self, x1, x2):
    if self.distance == "euclidean":
        return self.euclidean_distance(x1, x2)
    elif self.distance == "manhattan":
        return self.manhattan_distance(x1, x2)
    elif self.distance.startswith("minkowski"):
        # Example: distance="minkowski_3" → p=3
        p = int(self.distance.split("_")[1]) if "_" in self.distance else 3
        return self.minkowski_distance(x1, x2, p)
    else:
        raise ValueError(f"Unknown distance metric: {self.distance}")



 def fit(self , X , y) :
    self.X_train = X
    self.y_train = y

    #graphical representation of the model pipeline
    
    # ── Graphical representation of the model training ──────────────
   
    fig, ax = plt.subplots(figsize=(9, 11))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Header ──────────────────────────────────────────────────────
    header = FancyBboxPatch((0.12, 0.905), 0.76, 0.065,
        boxstyle="round,pad=0.01", facecolor="#1E293B",
        edgecolor="#38BDF8", linewidth=1.5, zorder=2)
    ax.add_patch(header)
    ax.text(0.5, 0.938, "LEARNML  |  KNN Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3)

    # ── Step definitions ─────────────────────────────────────────────
    steps = [
        {
            "y": 0.775,
            "title": "1. fit()  —  Memorize Training Data",
            # KNN has no real training — it just stores X and y
            "sub": "self.X_train = X     |     self.y_train = y",
            "color": "#38BDF8", "edge": "#0284C7",
        },
        {
            "y": 0.615,
            "title": "2. Compute Distances to Every Training Point",
            # distances = [self._compute_distance(x, x_train) for x_train in self.X_train]
            "sub": "distances = [dist(x, x_train)  for x_train in X_train]",
            "color": "#A855F7", "edge": "#7E22CE",
        },
        {
            "y": 0.455,
            "title": "3. Sort & Slice  —  Get the K Nearest",
            # np.argsort(distances) → indices in ascending order → [:k] = k closest
            "sub": "k_indices = np.argsort(distances)[:k]",
            "color": "#EC4899", "edge": "#BE185D",
        },
        {
            "y": 0.295,
            "title": "4. Collect Neighbor Labels",
            # k_nearest_labels = [self.y_train[i] for i in k_indices]
            "sub": "k_labels = [y_train[i]  for i in k_indices]",
            "color": "#F59E0B", "edge": "#B45309",
        },
        {
            "y": 0.115,
            "title": "5. Majority Vote  —  Return Prediction",
            # Fast path: np.bincount().argmax()  |  Fallback: Counter().most_common(1)
            "sub": "np.bincount(k_labels).argmax()   |   Counter(k_labels).most_common(1)",
            "color": "#10B981", "edge": "#059669",
        },
    ]

    card_w, card_h = 0.60, 0.095

    for step in steps:
        cy = step["y"]
        x0 = 0.5 - card_w / 2
        y0 = cy - card_h / 2

        # Card shadow
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.006, y0 - 0.006), card_w, card_h,
            boxstyle="round,pad=0.012", facecolor="#000000",
            edgecolor="none", alpha=0.4, zorder=2))

        # Card body
        ax.add_patch(FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.012", facecolor="#1E293B",
            edgecolor=step["edge"], linewidth=2, zorder=3))

        # Colored left accent bar
        ax.add_patch(FancyBboxPatch(
            (x0, y0), 0.012, card_h,
            boxstyle="round,pad=0.0", facecolor=step["color"],
            edgecolor="none", zorder=4))

        # Title
        ax.text(0.5, cy + 0.018, step["title"],
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=step["color"], fontfamily="sans-serif", zorder=5)

        # Formula / subtitle
        ax.text(0.5, cy - 0.022, step["sub"],
            ha="center", va="center", fontsize=9,
            color="#94A3B8", fontfamily="monospace", zorder=5)

    # ── Vertical arrows ──────────────────────────────────────────────
    gap = 0.012
    for i in range(len(steps) - 1):
        y_start = steps[i]["y"]   - card_h / 2 - gap
        y_end   = steps[i+1]["y"] + card_h / 2 + gap
        ax.annotate("", xy=(0.5, y_end), xytext=(0.5, y_start),
            arrowprops=dict(arrowstyle="-|>", color="#64748B",
                            lw=1.8, mutation_scale=16), zorder=3)

    # ── Per-query-point loop arc (Step 5 → Step 2) ───────────────────
    # predict() calls predict_one(x) for every x in X
    # so steps 2 → 3 → 4 → 5 repeat for each new query point
    verts = [
        (0.81, steps[4]["y"]),   # start: right edge of step 5
        (0.96, steps[2]["y"]),   # bezier control point
        (0.81, steps[1]["y"]),   # end: right edge of step 2
    ]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(mpatches.PathPatch(
        Path(verts, codes), facecolor="none",
        edgecolor="#F59E0B", linewidth=2,
        linestyle=(0, (5, 4)), zorder=3))

    ax.annotate("", xy=(0.812, steps[1]["y"]),
        xytext=(0.83, steps[1]["y"] + 0.03),
        arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                        lw=1.5, mutation_scale=14), zorder=4)

    ax.text(0.965, steps[2]["y"] + 0.025, "Per query\npoint",
        ha="left", va="center", fontsize=9.5, fontweight="bold",
        color="#F59E0B", fontfamily="monospace", zorder=5)
    ax.text(0.965, steps[2]["y"] - 0.025, "repeats for\neach x in X",
        ha="left", va="center", fontsize=8.5,
        color="#64748B", fontfamily="monospace", zorder=5)

    plt.tight_layout()
    plt.show()


 def predict(self , X):
   predictions = np.array([self.predict_one(x) for x in X])
   return predictions
 

 def predict_one(self , x):
  #compute the distance
  distances = [self._compute_distance(x , x_train) for x_train in self.X_train]

  #get the closest k
  k_indices = np.argsort(distances)[:self.k] # Get the indices of the k nearest points via slicing:
# np.argsort(distances) sorts all distances and returns their indices in ascending order.
# [:self.k] slices the first k indices, giving us the k closest neighbors
  
#compare to y   
  k_nearest_labels = [self.y_train[i] for i in k_indices]


# Majority vote
  try:
      # Fast path: works if labels are integers
      most_common_label = np.bincount(k_nearest_labels).argmax()
  except Exception:
      # Fallback: works for strings or mixed labels
      most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
  return most_common_label




  













    