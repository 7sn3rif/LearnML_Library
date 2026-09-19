import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.path as Path




def sigmoid(x):
  return 1/(1+np.exp(-x))






class LogisticRegression():
  def __init__(self, learning_rate=0.01, n_iters=100000):
    self.learning_rate=learning_rate
    self.n_iters = n_iters
   
    self.weights = None
    self.bias = None

  


  def fit(self, X, y):
     #Model training :
    
    n_samples, n_features = X.shape #features are the number of columns in the input data, and samples are the number of rows in the input data
    self.weights = np.zeros(n_features)
    self.bias = 0
    for _ in range(self.n_iters):
      y_predicted = sigmoid(np.dot(X, self.weights) + self.bias)
       #minimize the loss function using gradient descent
       #The error for each parameter (weight or bias) in logistic regression is indeed the derivative of the loss function (cross entropy) with respect to that parameter
      dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
      db = (1/n_samples) * np.sum(y_predicted - y)
      #update parameters
      self.weights = self.weights - self.learning_rate * dw
      self.bias = self.bias - self.learning_rate * db
           
        
    
    
    # ── Figure Setup ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 11))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    # ── Header ────────────────────────────────────────────────────────
    header = FancyBboxPatch((0.12, 0.905), 0.76, 0.065,
        boxstyle="round,pad=0.01", facecolor="#1E293B",
        edgecolor="#38BDF8", linewidth=1.5, zorder=2)
    ax.add_patch(header)
    ax.text(0.5, 0.938, "LEARNML  |  Logistic Regression Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3)
    
    # ── Steps ─────────────────────────────────────────────────────────
    steps = [
        {
            "y": 0.775,
            "title": "1. Initialize Parameters",
            # self.weights = np.zeros(n_features)  |  self.bias = 0
            "sub": "self.weights = np.zeros(n_features)     self.bias = 0",
            "color": "#38BDF8", "edge": "#0284C7",
        },
        {
            "y": 0.615,
            "title": "2. Sigmoid — Squash to Probability",
            # y_predicted = sigmoid(np.dot(X, self.weights) + self.bias)
            # sigmoid returns values between 0 and 1 → probability of class 1
            "sub": "y_pred = sigmoid( X · w + b )     →     σ(z) = 1 / (1 + e⁻ᶻ)",
            "color": "#A855F7", "edge": "#7E22CE",
        },
        {
            "y": 0.455,
            "title": "3. Compute Cross-Entropy Loss",
            # loss = -(1/n) * sum( y*log(y_pred) + (1-y)*log(1-y_pred) )
            # measures how far predicted probabilities are from true labels
            "sub": "J = -(1/m) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]",
            "color": "#EC4899", "edge": "#BE185D",
        },
        {
            "y": 0.295,
            "title": "4. Compute Gradients & Update",
            # dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/n_samples) * np.sum(y_predicted - y)
            # same gradient form as linear regression — cross-entropy makes it clean
            "sub": "dw = (1/m)·Xᵀ(ŷ−y)     db = (1/m)·Σ(ŷ−y)     w -= α·dw",
            "color": "#F59E0B", "edge": "#B45309",
        },
        {
            "y": 0.115,
            "title": "5. Training Complete",
            # after n_iters the weights have converged
            # predict() will now use sigmoid + threshold at 0.5 → class 0 or 1
            "sub": "Optimal weights saved — ready for  predict(X)",
            "color": "#10B981", "edge": "#059669",
        },
    ]
    
    card_w, card_h = 0.60, 0.095
    
    for step in steps:
        cy = step["y"]
        x0 = 0.5 - card_w / 2
        y0 = cy - card_h / 2
    
        # Shadow
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.006, y0 - 0.006), card_w, card_h,
            boxstyle="round,pad=0.012", facecolor="#000000",
            edgecolor="none", alpha=0.4, zorder=2))
    
        # Card body
        ax.add_patch(FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.012", facecolor="#1E293B",
            edgecolor=step["edge"], linewidth=2, zorder=3))
    
        # Left accent bar
        ax.add_patch(FancyBboxPatch(
            (x0, y0), 0.012, card_h,
            boxstyle="round,pad=0.0", facecolor=step["color"],
            edgecolor="none", zorder=4))
    
        # Title
        ax.text(0.5, cy + 0.018, step["title"],
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=step["color"], fontfamily="sans-serif", zorder=5)
    
        # Subtitle
        ax.text(0.5, cy - 0.022, step["sub"],
            ha="center", va="center", fontsize=9,
            color="#94A3B8", fontfamily="monospace", zorder=5)
    
    # ── Vertical arrows ───────────────────────────────────────────────
    gap = 0.012
    for i in range(len(steps) - 1):
        y_start = steps[i]["y"]   - card_h / 2 - gap
        y_end   = steps[i+1]["y"] + card_h / 2 + gap
        ax.annotate("", xy=(0.5, y_end), xytext=(0.5, y_start),
            arrowprops=dict(arrowstyle="-|>", color="#64748B",
                            lw=1.8, mutation_scale=16), zorder=3)
    
    # ── Gradient descent loop arc (Step 4 → Step 2) ──────────────────
    # for _ in range(n_iters): repeats sigmoid → loss → gradients → update
    verts = [
        (0.81, steps[3]["y"]),
        (0.96, steps[2]["y"]),
        (0.81, steps[1]["y"]),
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
    
    ax.text(0.965, steps[2]["y"] + 0.025, "Epoch\nLoop",
        ha="left", va="center", fontsize=9.5, fontweight="bold",
        color="#F59E0B", fontfamily="monospace", zorder=5)
    ax.text(0.965, steps[2]["y"] - 0.025, f"x {self.n_iters}\niters",
        ha="left", va="center", fontsize=8.5,
        color="#64748B", fontfamily="monospace", zorder=5)
    
    plt.tight_layout()
    plt.show()






def predict(self,X):
    y_predicted = sigmoid(np.dot(X, self.weights) + self.bias)   
    #the sigmoid fuctions return only values between 0 to 1
    
    #we need to make it return classes so :
    class_predicted = np.where(y_predicted > 0.5, 1, 0)
    return class_predicted
    #this function will return 1 if the predicted value is greater than 0.5 and 0 otherwise, effectively converting the continuous output of the sigmoid function into binary class labels.