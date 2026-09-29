
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import Arc

class NaiveBayes:
  def __init__(self) -> None:
    pass

  def fit(self, X ,y):
    n_samples, n_features = X.shape
    self._classes = np.unique(y)
    #get the number of unique classes
    n_classes = len(self._classes )
    #initialize the mean , the var and prior
    self._mean = np.zeros((n_classes , n_features), dtype=np.float64)
    self._var = np.zeros((n_classes , n_features), dtype=np.float64)
    self._priors = np.zeros(n_classes, dtype=np.float64) #prior is the probability of a class before looking at the features X
    #calculate the mean , the var and prior matricies
    for class_idx , c in enumerate(self._classes):
      X_c = X[y==c]
      self._mean[class_idx , :] = X_c.mean(axis=0)
      self._var[class_idx , :] = X_c.var(axis=0)
      self._priors[class_idx ] = X_c.shape[0] / float(n_samples)

   # ================================================================
   # Graphical representation of Gaussian Naive Bayes model training
   # ================================================================
   
    fig, ax = plt.subplots(figsize=(9, 11))
   
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
 
    # ── Header ───────────────────────────────────────────────────────
    
    header = FancyBboxPatch(
        (0.12, 0.905), 0.76, 0.065,
        boxstyle="round,pad=0.01",
        facecolor="#1E293B",
        edgecolor="#38BDF8",
        linewidth=1.5,
        zorder=2
    )
    
    ax.add_patch(header)
    
    ax.text(
        0.5, 0.938,
        "LEARNML  |  Gaussian Naive Bayes Training Pipeline",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace",
        zorder=3
    )
    
    
    # ── Step definitions ─────────────────────────────────────────────
    #
    # Gaussian Naive Bayes training:
    #
    # 1. Initialize parameters
    # 2. Find unique classes
    # 3. Separate data by class
    # 4. Calculate mean, variance and prior
    # 5. Training complete
    #
    # Step 4 is repeated for every class.
    
    steps = [
    
        {
            "y": 0.775,
            "title": "1. Initialize Parameters",
            "sub": "_mean = zeros()  |  _var = zeros()",
            "color": "#38BDF8",
            "edge": "#0284C7"
        },
    
        {
            "y": 0.615,
            "title": "2. Find Unique Classes",
            "sub": "_classes = np.unique(y)",
            "color": "#A855F7",
            "edge": "#7E22CE"
        },
    
        {
            "y": 0.455,
            "title": "3. Separate Samples by Class",
            "sub": "X_c = X[y == c]",
            "color": "#EC4899",
            "edge": "#BE185D"
        },
    
        {
            "y": 0.295,
            "title": "4. Calculate Class Statistics",
            "sub": "mean = X_c.mean()  |  var = X_c.var()  |  prior = n_c / n",
            "color": "#F59E0B",
            "edge": "#B45309"
        },
    
        {
            "y": 0.115,
            "title": "5. Training Complete",
            "sub": "Class means, variances & priors learned",
            "color": "#10B981",
            "edge": "#059669"
        },
    ]
    
    
    # ── Card dimensions ──────────────────────────────────────────────
    
    card_w = 0.60
    card_h = 0.095
    
    
    # ── Draw cards ───────────────────────────────────────────────────
    
    for step in steps:
    
        cy = step["y"]
    
        x0 = 0.5 - card_w / 2
        y0 = cy - card_h / 2
    
    
        # Card shadow
        shadow = FancyBboxPatch(
            (x0 + 0.006, y0 - 0.006),
            card_w,
            card_h,
            boxstyle="round,pad=0.012",
            facecolor="#000000",
            edgecolor="none",
            alpha=0.4,
            zorder=2
        )
    
        ax.add_patch(shadow)
    
    
        # Card body
        card = FancyBboxPatch(
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
    
    
        # Colored left accent bar
        accent = FancyBboxPatch(
            (x0, y0),
            0.012,
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
            cy + 0.018,
            step["title"],
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=step["color"],
            fontfamily="sans-serif",
            zorder=5
        )
    
    
        # Formula / subtitle
        ax.text(
            0.5,
            cy - 0.022,
            step["sub"],
            ha="center",
            va="center",
            fontsize=9.3,
            color="#94A3B8",
            fontfamily="monospace",
            zorder=5
        )
    
    
    # ── Vertical arrows ──────────────────────────────────────────────
    
    gap = 0.012
    
    arrow_pairs = [
        (
            steps[i]["y"] - card_h / 2 - gap,
            steps[i + 1]["y"] + card_h / 2 + gap
        )
        for i in range(len(steps) - 1)
    ]
    
    
    for y_start, y_end in arrow_pairs:
    
        ax.annotate(
            "",
            xy=(0.5, y_end),
            xytext=(0.5, y_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#64748B",
                lw=1.8,
                mutation_scale=16
            ),
            zorder=3
        )
    
    
    # ── Class loop: Step 4 → Step 3 ─────────────────────────────────
    #
    # For every class:
    #
    #     X_c = X[y == c]
    #
    #     mean = X_c.mean(axis=0)
    #     var  = X_c.var(axis=0)
    #     prior = X_c.shape[0] / n_samples
    #
    # Then move to the next class.
    #
    # Example:
    #
    # Class 0 → calculate statistics
    # Class 1 → calculate statistics
    # Class 2 → calculate statistics
    # ...
    #
    # until every class has been processed.
    
    
    loop_x = [0.81, 0.96, 0.81]
    
    loop_y = [
        steps[3]["y"],   # Step 4
        0.375,           # Middle of the loop
        steps[2]["y"]    # Step 3
    ]
    
    
    verts = [
        (loop_x[0], loop_y[0]),
        (loop_x[1], loop_y[1]),
        (loop_x[2], loop_y[2])
    ]
    
    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3
    ]
    
    path = Path(verts, codes)
    
    
    patch = mpatches.PathPatch(
        path,
        facecolor="none",
        edgecolor="#F59E0B",
        linewidth=2,
        linestyle=(0, (5, 4)),
        zorder=3
    )
    
    ax.add_patch(patch)
    
    
    # ── Arrowhead at end of class loop ───────────────────────────────
    
    ax.annotate(
        "",
        xy=(0.812, steps[2]["y"]),
        xytext=(0.83, steps[2]["y"] - 0.03),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#F59E0B",
            lw=1.5,
            mutation_scale=14
        ),
        zorder=4
    )
    
    
    # ── Loop label ───────────────────────────────────────────────────
    
    ax.text(
        0.965,
        0.375,
        "For each\nclass",
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#F59E0B",
        fontfamily="monospace",
        zorder=5
    )
    
    ax.text(
        0.965,
        0.335,
        "calculate\nstatistics",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#64748B",
        fontfamily="monospace",
        zorder=5
    )
    
    
    # ── Small explanatory labels ─────────────────────────────────────
    #
    # These make the diagram easier to understand without changing
    # the overall visual style.
    
    ax.text(
        0.16,
        0.295,
        "μ",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#F59E0B",
        fontfamily="monospace",
        zorder=5
    )
    
    ax.text(
        0.16,
        0.265,
        "mean",
        ha="center",
        va="center",
        fontsize=8,
        color="#64748B",
        fontfamily="monospace",
        zorder=5
    )
    
    ax.text(
        0.84,
        0.295,
        "σ²",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#F59E0B",
        fontfamily="monospace",
        zorder=5
    )
    
    ax.text(
        0.84,
        0.265,
        "variance",
        ha="center",
        va="center",
        fontsize=8,
        color="#64748B",
        fontfamily="monospace",
        zorder=5
    )
    
    
    # ── Finish ───────────────────────────────────────────────────────
    
    plt.tight_layout()
    
    plt.savefig(
        "naive_bayes_training_pipeline.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0F172A"
    )
    plt.show()
    
  def _pdf(self, x, class_idx):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
  def _predict(self, x):
        posteriors = []
        for idx , c in enumerate(self._classes):
          prior = np.log(self._priors[idx])
          posterior = np.sum(np.log(self._pdf(x, idx)))
          posterior = prior + posterior
          posteriors.append(posterior) 
        return self._classes[np.argmax(posteriors)]
   
  def predict(self, X):
       y_pred = [self._predict(x) for x in X]
       return np.array(y_pred)