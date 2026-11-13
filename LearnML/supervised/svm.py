import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.path import Path

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # Convert labels from {0, 1} to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b)
                if condition >= 1:
                    # Point is correctly classified and outside margin strip
                    # Apply only L2 regularization gradient to shrink weights
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Point violates the margin or is misclassified
                    # Update weights using regularization and hinge-loss gradients
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - y_[idx] * x_i)
                    self.b += self.learning_rate * y_[idx]

        # ── Graphical representation of the Linear SVM training pipeline ──
        fig, ax = plt.subplots(figsize=(9, 11))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        
        # ── Header ──────────────────────────────────────────────────────────
        header = FancyBboxPatch((0.12, 0.905), 0.76, 0.065,
            boxstyle="round,pad=0.01", facecolor="#1E293B",
            edgecolor="#38BDF8", linewidth=1.5, zorder=2)
        ax.add_patch(header)
        ax.text(0.5, 0.938, "LEARNML  |  Linear SVM Primal Training Pipeline",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#F8FAFC", fontfamily="monospace", zorder=3)
        
        # ── Step definitions ─────────────────────────────────────────────────
        steps = [
            {"y": 0.775, "title": "1. Initialize Parameters",
             "sub": "w = zeros(n_features)  |  b = 0  |  y_ in {-1, 1}",
             "color": "#38BDF8", "edge": "#0284C7"},
            {"y": 0.615, "title": "2. Iterate Epochs & Samples",
             "sub": "for epoch in range(n_iters): for idx, x_i in enumerate(X)",
             "color": "#A855F7", "edge": "#7E22CE"},
            {"y": 0.455, "title": "3. Evaluate Margin Condition",
             "sub": "condition = y_idx * (dot(x_i, w) + b)",
             "color": "#EC4899", "edge": "#BE185D"},
            {"y": 0.295, "title": "4. Gradient Descent Update",
             "sub": "if >=1: L2 shrinkage  |  else: Hinge loss + bias update",
             "color": "#F59E0B", "edge": "#B45309"},
            {"y": 0.115, "title": "5. Converged Model Weights",
             "sub": "Return optimal weight vector self.w and bias self.b",
             "color": "#10B981", "edge": "#059669"},
        ]
        
        card_w, card_h = 0.60, 0.095
        
        for step in steps:
            cy = step["y"]
            x0 = 0.5 - card_w / 2
            y0 = cy - card_h / 2
        
            # Card shadow
            shadow = FancyBboxPatch((x0 + 0.006, y0 - 0.006), card_w, card_h,
                boxstyle="round,pad=0.012", facecolor="#000000",
                edgecolor="none", alpha=0.4, zorder=2)
            ax.add_patch(shadow)
        
            # Card body
            card = FancyBboxPatch((x0, y0), card_w, card_h,
                boxstyle="round,pad=0.012", facecolor="#1E293B",
                edgecolor=step["edge"], linewidth=2, zorder=3)
            ax.add_patch(card)
        
            # Colored left accent bar
            accent = FancyBboxPatch((x0, y0), 0.012, card_h,
                boxstyle="round,pad=0.0", facecolor=step["color"],
                edgecolor="none", zorder=4)
            ax.add_patch(accent)
        
            # Title
            ax.text(0.5, cy + 0.018, step["title"],
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=step["color"], fontfamily="sans-serif", zorder=5)
        
            # Formula / subtitle
            ax.text(0.5, cy - 0.022, step["sub"],
                ha="center", va="center", fontsize=10,
                color="#94A3B8", fontfamily="monospace", zorder=5)
        
        # ── Vertical arrows ──────────────────────────────────────────────────
        gap = 0.012
        arrow_pairs = [
            (steps[i]["y"] - card_h / 2 - gap, steps[i+1]["y"] + card_h / 2 + gap)
            for i in range(len(steps) - 1)
        ]
        
        for y_start, y_end in arrow_pairs:
            ax.annotate("", xy=(0.5, y_end), xytext=(0.5, y_start),
                arrowprops=dict(arrowstyle="-|>", color="#64748B",
                                lw=1.8, mutation_scale=16),
                zorder=3)
        
        # ── Iterative loop arc (Step 4 → Step 2) ────────────────────────────
        loop_x = [0.81, 0.96, 0.81]
        loop_y = [steps[3]["y"], 0.455, steps[1]["y"]]
        
        verts = [
            (loop_x[0], loop_y[0]),
            (loop_x[1], loop_y[1]),
            (loop_x[2], loop_y[2]),
        ]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        patch = mpatches.PathPatch(path, facecolor="none",
            edgecolor="#F59E0B", linewidth=2,
            linestyle=(0, (5, 4)), zorder=3)
        ax.add_patch(patch)
        
        # Arrowhead at end of arc pointing back to Step 2
        ax.annotate("", xy=(0.812, steps[1]["y"]), xytext=(0.83, steps[1]["y"] + 0.03),
            arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                            lw=1.5, mutation_scale=14), zorder=4)
        
        # Loop label
        ax.text(0.965, 0.455, "Epoch\nLoop",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color="#F59E0B", fontfamily="monospace", zorder=5)
        ax.text(0.965, 0.415, "for n_iters",
            ha="left", va="center", fontsize=8.5,
            color="#64748B", fontfamily="monospace", zorder=5)
        
        plt.tight_layout()
        plt.savefig("linear_svm_pipeline.png", dpi=150, bbox_inches="tight",
                    facecolor="#0F172A")
        plt.show()

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)