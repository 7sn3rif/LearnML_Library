import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path


def learn_about_LinearRegression():
    
  # ── Learn About Linear Regression ────────────────────────────────
  print("Linear Regression is a supervised learning algorithm used for predicting a continuous target variable based on one or more input features. It assumes a linear relationship between the input features and the target variable. The goal of linear regression is to find the best-fitting line (or hyperplane in higher dimensions) that minimizes the difference between the predicted values and the actual values of the target variable.")
  print("\nKey Concepts:")
  print("1. Model Representation: The linear regression model can be represented as: y = w1*x1 + w2*x2 + ... + wn*xn + b, where y is the predicted output, x1, x2, ..., xn are the input features, w1, w2, ..., wn are the weights (coefficients), and b is the bias (intercept).")
  print("2. Loss Function: The most commonly used loss function for linear regression is Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values.")
  print("3. Optimization: The model parameters (weights and bias) are optimized using techniques like Gradient Descent to minimize the loss function.")
  print("4. Assumptions: Linear regression assumes linearity, independence of errors, homoscedasticity (constant variance of errors), and normality of error terms.")
  print("\nApplications:")
  print("- Predicting house prices based on features like size, location, and number of bedrooms.")
  print("- Forecasting sales based on advertising spend and other factors.")
  print("- Estimating demand for products based on historical data.")
  
  # ── Figure Setup ──────────────────────────────────────────────────
  fig, ax = plt.subplots(figsize=(10, 13))
  fig.patch.set_facecolor("#0F172A")
  ax.set_facecolor("#0F172A")
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis("off")
  
  # ── Header ────────────────────────────────────────────────────────
  header = mpatches.FancyBboxPatch(
      (0.12, 0.905), 0.76, 0.065,
      boxstyle="round,pad=0.01", facecolor="#1E293B",
      edgecolor="#38BDF8", linewidth=1.5, zorder=2)
  ax.add_patch(header)
  
  ax.text(0.5, 0.938, "LEARNML  |  How Linear Regression works",
      ha="center", va="center", fontsize=13, fontweight="bold",
      color="#F8FAFC", fontfamily="monospace", zorder=3)
  
  # ── Hyperparameter badges ─────────────────────────────────────────
  weights       = None
  n_iters       = 1000
  learning_rate = 0.001
  
  fitted   = weights is not None
  lr_text  = f"lr = {learning_rate}"
  itr_text = f"iters = {n_iters}"
  st_text  = "fitted \u2713" if fitted else "not fitted"
  st_color = "#10B981" if fitted else "#F59E0B"
  
  for i, (label, color) in enumerate([
      (lr_text,  "#38BDF8"),
      (itr_text, "#A855F7"),
      (st_text,  st_color),
  ]):
      bx = 0.10 + i * 0.295
      ax.add_patch(mpatches.FancyBboxPatch(
          (bx, 0.856), 0.265, 0.038,
          boxstyle="round,pad=0.010", facecolor="#1E293B",
          edgecolor=color, linewidth=1.2, zorder=2))
      ax.text(bx + 0.1325, 0.875, label,
          ha="center", va="center", fontsize=9.5,
          color=color, fontfamily="monospace", zorder=3)
  
  # ── Steps ─────────────────────────────────────────────────────────
  steps = [
      {
          "y": 0.770,
          "icon": "w,b",
          "title": "Initialize",
          "idea": "Start every weight at zero and bias at zero. The model knows nothing yet — training will shape it.",
          "math": "w = [0, 0, ..., 0]     b = 0",
          "color": "#38BDF8", "edge": "#0284C7",
      },
      {
          "y": 0.618,
          "icon": "y^",
          "title": "Predict",
          "idea": "Multiply each feature by its weight and sum them up. Add the bias. This is the model's current guess.",
          "math": "y_pred = X . w + b",
          "color": "#A855F7", "edge": "#7E22CE",
      },
      {
          "y": 0.466,
          "icon": "J",
          "title": "Measure Error",
          "idea": "Compare every prediction to the true value. Square the differences so big errors hurt more. Average them.",
          "math": "J = (1 / 2m) . sum( (y_pred - y)^2 )",
          "color": "#EC4899", "edge": "#BE185D",
      },
      {
          "y": 0.314,
          "icon": "dJ",
          "title": "Compute Gradients",
          "idea": "Ask: which direction does each weight need to move to reduce the error? The gradient answers that.",
          "math": "dw = (1/m) . X^T(y_pred - y)     db = (1/m) . sum(y_pred - y)",
          "color": "#F59E0B", "edge": "#B45309",
      },
      {
          "y": 0.162,
          "icon": "w-",
          "title": "Update Weights",
          "idea": "Take a small step opposite to the gradient. Learning rate controls how large that step is.",
          "math": "w  <-  w - lr . dw          b  <-  b - lr . db",
          "color": "#FB923C", "edge": "#C2410C",
      },
  ]
  
  card_w = 0.68
  card_h = 0.112
  gap    = 0.010
  icon_r = 0.038
  
  for step in steps:
      cy = step["y"]
      x0 = 0.5 - card_w / 2
      y0 = cy - card_h / 2
  
      # Shadow
      ax.add_patch(mpatches.FancyBboxPatch(
          (x0 + 0.007, y0 - 0.007), card_w, card_h,
          boxstyle="round,pad=0.014", facecolor="#000000",
          edgecolor="none", alpha=0.35, zorder=2))
  
      # Card
      ax.add_patch(mpatches.FancyBboxPatch(
          (x0, y0), card_w, card_h,
          boxstyle="round,pad=0.014", facecolor="#1E293B",
          edgecolor=step["edge"], linewidth=1.8, zorder=3))
  
      # Icon circle
      icon_x = x0 + 0.056
      ax.add_patch(plt.Circle(
          (icon_x, cy), icon_r,
          facecolor=step["edge"], zorder=4))
      ax.text(icon_x, cy, step["icon"],
          ha="center", va="center", fontsize=9, fontweight="bold",
          color="#F8FAFC", fontfamily="monospace", zorder=5)
  
      # Title
      ax.text(x0 + 0.115, cy + 0.034, step["title"],
          ha="left", va="center", fontsize=12, fontweight="bold",
          color=step["color"], fontfamily="sans-serif", zorder=5)
  
      # Plain English
      ax.text(x0 + 0.115, cy + 0.006, step["idea"],
          ha="left", va="center", fontsize=9, color="#CBD5E1",
          fontfamily="sans-serif", zorder=5)
  
      # Divider
      ax.plot(
          [x0 + 0.115, x0 + card_w - 0.018],
          [cy - 0.018, cy - 0.018],
          color="#2D3748", lw=0.8, zorder=4)
  
      # Math
      ax.text(x0 + 0.115, cy - 0.036, step["math"],
          ha="left", va="center", fontsize=9.5,
          color="#64748B", fontfamily="monospace", zorder=5)
  
  # ── Vertical arrows ───────────────────────────────────────────────
  for i in range(len(steps) - 1):
      y_start = steps[i]["y"]     - card_h / 2 - gap
      y_end   = steps[i + 1]["y"] + card_h / 2 + gap
      ax.annotate("", xy=(0.5, y_end), xytext=(0.5, y_start),
          arrowprops=dict(arrowstyle="-|>", color="#475569",
                          lw=1.8, mutation_scale=16), zorder=3)
  
  # ── Gradient descent loop arc (Step 5 → Step 2) ──────────────────
  s_start = steps[4]["y"]
  s_mid   = steps[2]["y"]
  s_end   = steps[1]["y"]
  arc_x   = 0.87
  
  verts = [(arc_x, s_start), (arc_x + 0.08, s_mid), (arc_x, s_end)]
  codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
  ax.add_patch(mpatches.PathPatch(
      Path(verts, codes), facecolor="none",
      edgecolor="#F59E0B", linewidth=2,
      linestyle=(0, (5, 4)), zorder=3))
  
  ax.annotate("", xy=(arc_x + 0.002, s_end),
      xytext=(arc_x + 0.022, s_end + 0.038),
      arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                      lw=1.5, mutation_scale=13), zorder=4)
  
  mid_y = (s_start + s_end) / 2
  ax.text(arc_x + 0.090, mid_y + 0.022, "Repeat each\nepoch",
      ha="left", va="center", fontsize=9, fontweight="bold",
      color="#F59E0B", fontfamily="sans-serif", zorder=5)
  ax.text(arc_x + 0.090, mid_y - 0.030, f"x {n_iters}\niterations",
      ha="left", va="center", fontsize=8.5,
      color="#64748B", fontfamily="sans-serif", zorder=5)
  
  # ── Convergence note ──────────────────────────────────────────────
  status = (
      f"Fitted  —  weights shape: {list(weights.shape)}    bias: {0:.4f}"
      if fitted else
      "Call  model.fit(X, y)  to train the model"
  )
  color = "#10B981" if fitted else "#F59E0B"
  edge  = "#059669" if fitted else "#B45309"
  bg    = "#052e16" if fitted else "#1c1008"
  
  ax.add_patch(mpatches.FancyBboxPatch(
      (0.08, 0.022), 0.84, 0.052,
      boxstyle="round,pad=0.010", facecolor=bg,
      edgecolor=edge, linewidth=1.2, zorder=2))
  ax.text(0.5, 0.048, status,
      ha="center", va="center", fontsize=10,
      color=color, fontfamily="monospace", zorder=3)
  
  plt.tight_layout()
  plt.show()



def learn_about_PolynomialRegression():
    print("Polynomial Regression is an extension of linear regression that allows for modeling non-linear relationships between the input features and the target variable. It does this by introducing polynomial terms (e.g., x^2, x^3) into the regression equation, enabling the model to fit more complex curves to the data.")
    print("\nKey Concepts:")
    print("1. Model Representation: The polynomial regression model can be represented as: y = w0 + w1*x + w2*x^2 + ... + wn*x^n, where y is the predicted output, x is the input feature, and w0, w1, ..., wn are the weights (coefficients).")
    print("2. Degree of Polynomial: The degree of the polynomial determines the flexibility of the model. A higher degree allows for more complex curves but may lead to overfitting.")
    print("3. Feature Transformation: Polynomial regression involves transforming the original features into polynomial features before fitting the model.")
    print("\nApplications:")
    print("- Modeling growth rates in biology or economics.")
    print("- Predicting trends in stock prices or market data.")
    print("- Fitting curves to experimental data in physics or chemistry.")

    

    # ── State (mirror self.* when inside the class) ───────────────
    weights       = None
    degree        = 2
    n_iters       = 1000
    learning_rate = 0.001

    fitted   = weights is not None
    lr_text  = f"lr = {learning_rate}"
    itr_text = f"iters = {n_iters}"
    deg_text = f"degree = {degree}"
    st_text  = "fitted \u2713" if fitted else "not fitted"
    st_color = "#10B981" if fitted else "#F59E0B"

    # ── Figure Setup ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 13))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── Header ────────────────────────────────────────────────────
    header = mpatches.FancyBboxPatch(
        (0.12, 0.905), 0.76, 0.065,
        boxstyle="round,pad=0.01", facecolor="#1E293B",
        edgecolor="#A855F7", linewidth=1.5, zorder=2)
    ax.add_patch(header)
    ax.text(0.5, 0.938, "LEARNML  |  How Polynomial Regression works",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3)

    # ── Hyperparameter badges ─────────────────────────────────────
    for i, (label, color) in enumerate([
        (lr_text,  "#38BDF8"),
        (itr_text, "#A855F7"),
        (deg_text, "#EC4899"),
        (st_text,  st_color),
    ]):
        bx = 0.05 + i * 0.225
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, 0.856), 0.205, 0.038,
            boxstyle="round,pad=0.010", facecolor="#1E293B",
            edgecolor=color, linewidth=1.2, zorder=2))
        ax.text(bx + 0.1025, 0.875, label,
            ha="center", va="center", fontsize=9,
            color=color, fontfamily="monospace", zorder=3)

    # ── Steps ─────────────────────────────────────────────────────
    steps = [
        {
            "y": 0.770,
            "icon": "w,b",
            "title": "Initialize",
            "idea": "Set all weights to zero, bias to zero, and choose the polynomial degree. Nothing learned yet.",
            "math": "w = [0, ..., 0]     b = 0     degree = d",
            "color": "#38BDF8", "edge": "#0284C7",
        },
        {
            "y": 0.618,
            "icon": "x^d",
            "title": "Expand Features",
            "idea": "Transform raw input into polynomial features. Each degree adds a new column: x, x², x³ … xᵈ.",
            "math": "X_poly[:, d-1] = X.flatten() ** d     for d in 1..degree",
            "color": "#A855F7", "edge": "#7E22CE",
        },
        {
            "y": 0.466,
            "icon": "y^",
            "title": "Predict",
            "idea": "Run standard linear prediction on the expanded features. The curve comes from the poly columns.",
            "math": "y_pred = X_poly . w + b",
            "color": "#EC4899", "edge": "#BE185D",
        },
        {
            "y": 0.314,
            "icon": "J",
            "title": "Measure Error",
            "idea": "Compare predictions to true values using MSE. Squaring penalises large errors more heavily.",
            "math": "J = (1 / 2m) . sum( (y_pred - y)^2 )",
            "color": "#F59E0B", "edge": "#B45309",
        },
        {
            "y": 0.162,
            "icon": "w-",
            "title": "Update Weights",
            "idea": "Compute gradients over the poly features and nudge every weight one small step downhill.",
            "math": "w  <-  w - lr . dw          b  <-  b - lr . db",
            "color": "#FB923C", "edge": "#C2410C",
        },
    ]

    card_w = 0.68
    card_h = 0.112
    gap    = 0.010
    icon_r = 0.038

    for step in steps:
        cy = step["y"]
        x0 = 0.5 - card_w / 2
        y0 = cy - card_h / 2

        # Shadow
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0 + 0.007, y0 - 0.007), card_w, card_h,
            boxstyle="round,pad=0.014", facecolor="#000000",
            edgecolor="none", alpha=0.35, zorder=2))

        # Card
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.014", facecolor="#1E293B",
            edgecolor=step["edge"], linewidth=1.8, zorder=3))

        # Icon circle
        icon_x = x0 + 0.056
        ax.add_patch(plt.Circle(
            (icon_x, cy), icon_r,
            facecolor=step["edge"], zorder=4))
        ax.text(icon_x, cy, step["icon"],
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#F8FAFC", fontfamily="monospace", zorder=5)

        # Title
        ax.text(x0 + 0.115, cy + 0.034, step["title"],
            ha="left", va="center", fontsize=12, fontweight="bold",
            color=step["color"], fontfamily="sans-serif", zorder=5)

        # Plain English
        ax.text(x0 + 0.115, cy + 0.006, step["idea"],
            ha="left", va="center", fontsize=9, color="#CBD5E1",
            fontfamily="sans-serif", zorder=5)

        # Divider
        ax.plot(
            [x0 + 0.115, x0 + card_w - 0.018],
            [cy - 0.018, cy - 0.018],
            color="#2D3748", lw=0.8, zorder=4)

        # Math
        ax.text(x0 + 0.115, cy - 0.036, step["math"],
            ha="left", va="center", fontsize=9.5,
            color="#64748B", fontfamily="monospace", zorder=5)

    # ── Vertical arrows ───────────────────────────────────────────
    for i in range(len(steps) - 1):
        y_start = steps[i]["y"]     - card_h / 2 - gap
        y_end   = steps[i + 1]["y"] + card_h / 2 + gap
        ax.annotate("", xy=(0.5, y_end), xytext=(0.5, y_start),
            arrowprops=dict(arrowstyle="-|>", color="#475569",
                            lw=1.8, mutation_scale=16), zorder=3)

    # ── Gradient descent loop arc (Step 5 → Step 3) ──────────────
    s_start = steps[4]["y"]
    s_mid   = steps[3]["y"]
    s_end   = steps[2]["y"]
    arc_x   = 0.87

    verts = [(arc_x, s_start), (arc_x + 0.08, s_mid), (arc_x, s_end)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    ax.add_patch(mpatches.PathPatch(
        Path(verts, codes), facecolor="none",
        edgecolor="#F59E0B", linewidth=2,
        linestyle=(0, (5, 4)), zorder=3))

    ax.annotate("", xy=(arc_x + 0.002, s_end),
        xytext=(arc_x + 0.022, s_end + 0.038),
        arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                        lw=1.5, mutation_scale=13), zorder=4)

    mid_y = (s_start + s_end) / 2
    ax.text(arc_x + 0.090, mid_y + 0.022, "Repeat each\nepoch",
        ha="left", va="center", fontsize=9, fontweight="bold",
        color="#F59E0B", fontfamily="sans-serif", zorder=5)
    ax.text(arc_x + 0.090, mid_y - 0.030, f"x {n_iters}\niterations",
        ha="left", va="center", fontsize=8.5,
        color="#64748B", fontfamily="sans-serif", zorder=5)

    # ── Convergence note ──────────────────────────────────────────
    status = (
        f"Fitted  —  weights shape: {list(weights.shape)}    bias: {0:.4f}"
        if fitted else
        "Call  model.fit(X, y)  to train the model"
    )
    color = "#10B981" if fitted else "#F59E0B"
    edge  = "#059669" if fitted else "#B45309"
    bg    = "#052e16" if fitted else "#1c1008"

    ax.add_patch(mpatches.FancyBboxPatch(
        (0.08, 0.022), 0.84, 0.052,
        boxstyle="round,pad=0.010", facecolor=bg,
        edgecolor=edge, linewidth=1.2, zorder=2))
    ax.text(0.5, 0.048, status,
        ha="center", va="center", fontsize=10,
        color=color, fontfamily="monospace", zorder=3)

    plt.tight_layout()
    plt.show()
 