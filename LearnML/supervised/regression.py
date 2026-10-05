import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import Arc


class LinearRegression:
  def __init__(self, learning_rate=0.001, n_iters=1000):
    self.learning_rate  = learning_rate
    self.n_iters = n_iters
    self.weights = None
    self.bias = None


  def fit(self, X, y):
    #Model training
    n_samples, n_features = X.shape #features are the number of columns in the input data, and samples are the number of rows in the input data
    self.weights = np.zeros(n_features)
    self.bias = 0
    for _ in range(self.n_iters):
      y_predicted = np.dot(X, self.weights) + self.bias
      #minimize the loss function using gradient descent
      #The error for each parameter (weight or bias) in the model by taking the derivative of the loss function with respect to that parameter. The gradients indicate the direction and magnitude of change needed to minimize the loss.   
      dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
      db = (1/n_samples) * np.sum(y_predicted - y)
      #update parameters
      self.weights = self.weights - self.learning_rate * dw
      self.bias = self.bias - self.learning_rate * db
      #until here , the algorithm is a standard implementation of linear regression using gradient descent. and the rest of the code is for visualization purposes, and it is not part of the core algorithm.
      #the following code generates a graphical representation of the linear regression training pipeline, illustrating the steps involved in the process.

    
    #graphical representation of the model training      
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
    ax.text(0.5, 0.938, "LEARNML  |  Linear Regression Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3)
    
    # ── Step definitions ─────────────────────────────────────────────────
    steps = [
        {"y": 0.775, "title": "1. Initialize Parameters",
         "sub": "w = zeros()  |  b = 0.0",
         "color": "#38BDF8", "edge": "#0284C7"},
        {"y": 0.615, "title": "2. Forward Prediction",
         "sub": "y_pred = X · w + b",
         "color": "#A855F7", "edge": "#7E22CE"},
        {"y": 0.455, "title": "3. Compute MSE Loss",
         "sub": "J(w,b) = (1/2m) Σ (y_pred − y)²",
         "color": "#EC4899", "edge": "#BE185D"},
        {"y": 0.295, "title": "4. Compute Gradients & Update",
         "sub": "w -= α·∂J/∂w   |   b -= α·∂J/∂b",
         "color": "#F59E0B", "edge": "#B45309"},
        {"y": 0.115, "title": "5. Training Complete",
         "sub": "Model converged — optimal weights saved",
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
    
    # ── Epoch loop arc (Step 4 → Step 2) ────────────────────────────────
   
    
    # Dashed curved path via bezier control points
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
    
    # Arrowhead at end of arc
    ax.annotate("", xy=(0.812, steps[1]["y"]), xytext=(0.83, steps[1]["y"] + 0.03),
        arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                        lw=1.5, mutation_scale=14), zorder=4)
    
    # Loop label
    ax.text(0.965, 0.455, "Epoch\nLoop",
        ha="left", va="center", fontsize=9.5, fontweight="bold",
        color="#F59E0B", fontfamily="monospace", zorder=5)
    ax.text(0.965, 0.415, "until\nconvergence",
        ha="left", va="center", fontsize=8.5,
        color="#64748B", fontfamily="monospace", zorder=5)
    
    plt.tight_layout()
    plt.savefig("learnml_pipeline.png", dpi=150, bbox_inches="tight",
                facecolor="#0F172A")
    plt.show()

  def predict(self, X ):
    y_predicted = np.dot(X, self.weights) + self.bias
    return y_predicted














class PolynomialRegression:
    def __init__(self, degree=2, learning_rate=0.001, n_iters=1000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def _polynomial_features(self, X ):
        n_samples = X.shape[0] #to count how many samples(rows) are in the input data
        X_poly = np.ones((n_samples, self.degree))  # polynomial expansion , to create a new feature matrix with polynomial features up to the specified degree
        for d in range(1, self.degree + 1): # inside the loop, we compute the polynomial features for each degree from 1 to the specified degree. For each degree d, we raise the input feature X to the power of d and store it in the corresponding column of the X_poly matrix. This way, we create a new feature matrix that includes polynomial features up to the specified degree.
            X_poly[:, d-1] = X.flatten() ** d
        return X_poly
    
    def _linear_model(self,X,y):
        #Model training
            n_samples, n_features = X.shape #features are the number of columns in the input data, and samples are the number of rows in the input data
            self.weights = np.zeros(n_features)
            self.bias = 0
            for _ in range(self.n_iters):
              y_predicted = np.dot(X, self.weights) + self.bias
              #minimize the loss function using gradient descent
              #The error for each parameter (weight or bias) in the model by taking the derivative of the loss function with respect to that parameter. The gradients indicate the direction and magnitude of change needed to minimize the loss.   
              dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
              db = (1/n_samples) * np.sum(y_predicted - y)
              #update parameters
              self.weights = self.weights - self.learning_rate * dw
              self.bias = self.bias - self.learning_rate * db
    

    def fit(self, X, y):
        X_poly = self._polynomial_features(X)
        self._linear_model(X_poly, y) #same as linear regression, but we use the polynomial features instead of the original features. This allows us to fit a polynomial regression model to the data.
        #now, the model is trained using the polynomial features, and we can use it to make predictions on new data.
        #the rest of the code is for visualization purposes, and it is not part of the core algorithm.

        #graphical representation of the polynomial regression training pipeline, illustrating the steps involved in the process.
        fig, ax = plt.subplots(figsize=(9, 12))
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        
        # ── Header ───────────────────────────────────────────────────────────
        header = mpatches.FancyBboxPatch((0.12, 0.918), 0.76, 0.062,
            boxstyle="round,pad=0.012", facecolor="#1E293B",
            edgecolor="#A855F7", linewidth=1.8, zorder=2)
        ax.add_patch(header)
        ax.text(0.5, 0.949, "LEARNML  |  Polynomial Regression Pipeline",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#F8FAFC", fontfamily="monospace", zorder=3)
        
        # ── Steps ─────────────────────────────────────────────────────────────
        steps = [
            {
                "y": 0.830,
                "title": "1. Initialize Parameters",
                "sub": "w = zeros()  |  b = 0.0  |  degree = d",
                "color": "#38BDF8", "edge": "#0284C7",
            },
            {
                "y": 0.672,
                "title": "2. Polynomial Feature Expansion",
                "sub": "X_poly[:, d-1] = X.flatten() ** d   →   [x, x², x³, ..., xᵈ]",
                "color": "#A855F7", "edge": "#7E22CE",
            },
            {
                "y": 0.514,
                "title": "3. Forward Prediction",
                "sub": "y_pred = X_poly · w + b",
                "color": "#EC4899", "edge": "#BE185D",
            },
            {
                "y": 0.356,
                "title": "4. Compute MSE Loss",
                "sub": "J(w,b) = (1/2m) Σ (y_pred − y)²",
                "color": "#F59E0B", "edge": "#B45309",
            },
            {
                "y": 0.198,
                "title": "5. Compute Gradients & Update",
                "sub": "w -= α·∂J/∂w   |   b -= α·∂J/∂b",
                "color": "#FB923C", "edge": "#C2410C",
            },
            {
                "y": 0.062,
                "title": "6. Training Complete",
                "sub": "Polynomial model converged — optimal weights saved",
                "color": "#10B981", "edge": "#059669",
            },
        ]
        
        card_w, card_h = 0.64, 0.090
        gap = 0.011
        
        for step in steps:
            cy = step["y"]
            x0 = 0.5 - card_w / 2
            y0 = cy - card_h / 2
        
            # Shadow
            shadow = mpatches.FancyBboxPatch((x0 + 0.006, y0 - 0.006), card_w, card_h,
                boxstyle="round,pad=0.012", facecolor="#000000",
                edgecolor="none", alpha=0.35, zorder=2)
            ax.add_patch(shadow)
        
            # Card
            card = mpatches.FancyBboxPatch((x0, y0), card_w, card_h,
                boxstyle="round,pad=0.012", facecolor="#1E293B",
                edgecolor=step["edge"], linewidth=2, zorder=3)
            ax.add_patch(card)
        
            # Left accent bar
            accent = mpatches.FancyBboxPatch((x0, y0), 0.013, card_h,
                boxstyle="round,pad=0.0", facecolor=step["color"],
                edgecolor="none", zorder=4)
            ax.add_patch(accent)
        
            # Title
            ax.text(0.5, cy + 0.020, step["title"],
                ha="center", va="center", fontsize=11.5, fontweight="bold",
                color=step["color"], fontfamily="sans-serif", zorder=5)
        
            # Subtitle
            ax.text(0.5, cy - 0.022, step["sub"],
                ha="center", va="center", fontsize=9.5,
                color="#94A3B8", fontfamily="monospace", zorder=5)
        
        # ── Vertical arrows ───────────────────────────────────────────────────
        for i in range(len(steps) - 1):
            y_start = steps[i]["y"]   - card_h / 2 - gap
            y_end   = steps[i+1]["y"] + card_h / 2 + gap
            ax.annotate("", xy=(0.5, y_end), xytext=(0.5, y_start),
                arrowprops=dict(arrowstyle="-|>", color="#475569",
                                lw=1.8, mutation_scale=16), zorder=3)
        
        # ── Degree annotation on step 2 (highlight the key difference) ───────
        ax.text(0.845, steps[1]["y"] + 0.014, "NEW",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="#0F172A", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#A855F7",
                      edgecolor="none"), zorder=6)
        
        # ── Epoch loop arc (Step 5 → Step 3) ─────────────────────────────────
        verts = [
            (0.818, steps[4]["y"]),       # start: right edge of step 5
            (0.970, steps[3]["y"]),       # bezier control point
            (0.818, steps[2]["y"]),       # end: right edge of step 3
        ]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        loop_patch = mpatches.PathPatch(path, facecolor="none",
            edgecolor="#F59E0B", linewidth=2,
            linestyle=(0, (5, 4)), zorder=3)
        ax.add_patch(loop_patch)
        
        # Arrowhead at end of arc
        ax.annotate("", xy=(0.820, steps[2]["y"]),
            xytext=(0.845, steps[2]["y"] + 0.032),
            arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                            lw=1.5, mutation_scale=14), zorder=4)
        
        # Loop label
        ax.text(0.978, steps[3]["y"] + 0.030, "Epoch\nLoop",
            ha="left", va="center", fontsize=9.5, fontweight="bold",
            color="#F59E0B", fontfamily="monospace", zorder=5)
        ax.text(0.978, steps[3]["y"] - 0.020, "until\nconvergence",
            ha="left", va="center", fontsize=8.5,
            color="#64748B", fontfamily="monospace", zorder=5)
        
        # ── Degree loop arc (Step 2 internal: d = 1 → degree) ────────────────
        d_x = 0.175
        d_y_top = steps[1]["y"] + card_h / 2 - 0.005
        d_y_bot = steps[1]["y"] - card_h / 2 + 0.005
        
        verts2 = [
            (d_x + 0.01, d_y_top),
            (d_x - 0.09, steps[1]["y"]),
            (d_x + 0.01, d_y_bot),
        ]
        codes2 = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path2 = Path(verts2, codes2)
        degree_patch = mpatches.PathPatch(path2, facecolor="none",
            edgecolor="#A855F7", linewidth=1.5,
            linestyle=(0, (4, 3)), zorder=3)
        ax.add_patch(degree_patch)
        
        ax.annotate("", xy=(d_x + 0.008, d_y_bot),
            xytext=(d_x - 0.015, d_y_bot + 0.025),
            arrowprops=dict(arrowstyle="-|>", color="#A855F7",
                            lw=1.2, mutation_scale=12), zorder=4)
        
        ax.text(0.055, steps[1]["y"], "for d in\nrange(1,\ndegree+1)",
            ha="center", va="center", fontsize=8,
            color="#A855F7", fontfamily="monospace", zorder=5)
        
        plt.tight_layout()
        plt.savefig("polynomial_regression_pipeline.png", dpi=150,
                    bbox_inches="tight", facecolor="#0F172A")
        plt.show()
        
        
    def predict(self, X):
             #same as linear regression, but we use the polynomial features instead of the original features. This allows us to make predictions using the trained polynomial regression model.
             X_poly = self._polynomial_features(X)
             y_predicted = np.dot(X_poly, self.weights) + self.bias
             return y_predicted