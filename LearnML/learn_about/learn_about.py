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
 












def learn_about_KNN():

    # ── Learn About KNN ───────────────────────────────────────────────
        print("KNN (K-Nearest Neighbors) is a simple, non-parametric supervised learning algorithm that makes predictions by finding the K closest training points to a new input and taking a majority vote (classification) or average (regression). Unlike other algorithms, KNN has no training phase — it memorizes the data and does all the work at prediction time.")
        print("\nKey Concepts:")
        print("1. Distance Metric: KNN measures closeness using Euclidean, Manhattan, or Minkowski distance. The choice of metric affects which neighbors are selected.")
        print("2. K Value: K controls how many neighbors vote. Small K = flexible but noisy. Large K = smooth but less sensitive to local patterns.")
        print("3. No Training Phase: KNN just memorizes the data — fit() only stores X and y. Prediction is where all the computation happens.")
        print("4. Majority Vote: The most common label among the K neighbors wins. For ties, the first-found label is returned.")
        print("\nApplications:")
        print("- Classifying tumors as malignant or benign based on cell features.")
        print("- Recommending products based on similar users' purchase history.")
        print("- Detecting anomalies in network traffic by finding outlier points.")
       
        # ── Standalone values (mirror class attributes) ───────────────────
        k        = 3
        distance = "euclidean"
        fitted   = False
        
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
        ax.text(0.5, 0.938, "LEARNML  |  How KNN works",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#F8FAFC", fontfamily="monospace", zorder=3)
        
        # ── Hyperparameter badges ─────────────────────────────────────────
        k_text   = f"k = {k}"
        d_text   = f"dist = {distance}"
        st_text  = "fitted \u2713" if fitted else "not fitted"
        st_color = "#10B981" if fitted else "#F59E0B"
        
        for i, (label, color) in enumerate([
            (k_text,  "#38BDF8"),
            (d_text,  "#A855F7"),
            (st_text, st_color),
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
                "icon": "fit",
                "title": "Memorize",
                "idea": "No learning happens. KNN just stores the entire training set. fit() is just two lines of assignment.",
                "math": "X_train = X          y_train = y",
                "color": "#38BDF8", "edge": "#0284C7",
            },
            {
                "y": 0.618,
                "icon": "d()",
                "title": "Compute Distances",
                "idea": "For each new point x, compute its distance to every single training point using the chosen metric.",
                "math": "distances = [dist(x, x_train)  for x_train in X_train]",
                "color": "#A855F7", "edge": "#7E22CE",
            },
            {
                "y": 0.466,
                "icon": "↑k",
                "title": "Get K Nearest",
                "idea": "Sort all distances ascending and slice the first k indices — those are the k closest neighbors.",
                "math": "k_indices = np.argsort(distances)[:k]",
                "color": "#EC4899", "edge": "#BE185D",
            },
            {
                "y": 0.314,
                "icon": "y[]",
                "title": "Collect Labels",
                "idea": "Look up the true label of each of the k nearest neighbors from the stored training labels.",
                "math": "k_labels = [y_train[i]  for i in k_indices]",
                "color": "#F59E0B", "edge": "#B45309",
            },
            {
                "y": 0.162,
                "icon": "vote",
                "title": "Majority Vote",
                "idea": "The most common label among k neighbors wins. bincount for integers, Counter as fallback for strings.",
                "math": "np.bincount(k_labels).argmax()   |   Counter(k_labels).most_common(1)",
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
                ha="center", va="center", fontsize=8, fontweight="bold",
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
                ha="left", va="center", fontsize=9,
                color="#64748B", fontfamily="monospace", zorder=5)
        
        # ── Vertical arrows ───────────────────────────────────────────────
        for i in range(len(steps) - 1):
            y_start = steps[i]["y"]     - card_h / 2 - gap
            y_end   = steps[i + 1]["y"] + card_h / 2 + gap
            ax.annotate("", xy=(0.5, y_end), xytext=(0.5, y_start),
                arrowprops=dict(arrowstyle="-|>", color="#475569",
                                lw=1.8, mutation_scale=16), zorder=3)
        
        # ── Per-query-point loop arc (Step 5 → Step 2) ───────────────────
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
        ax.text(arc_x + 0.090, mid_y + 0.022, "Per query\npoint",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color="#F59E0B", fontfamily="sans-serif", zorder=5)
        ax.text(arc_x + 0.090, mid_y - 0.030, "x len(X)\nqueries",
            ha="left", va="center", fontsize=8.5,
            color="#64748B", fontfamily="sans-serif", zorder=5)
        
        # ── Status bar ────────────────────────────────────────────────────
        status = "Call  model.fit(X, y)  to load the training data"
        color  = "#F59E0B"
        edge   = "#B45309"
        bg     = "#1c1008"
        
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.08, 0.022), 0.84, 0.052,
            boxstyle="round,pad=0.010", facecolor=bg,
            edgecolor=edge, linewidth=1.2, zorder=2))
        ax.text(0.5, 0.048, status,
            ha="center", va="center", fontsize=10,
            color=color, fontfamily="monospace", zorder=3)
        
        plt.tight_layout()
        plt.show()
           
            
        
        
        
        



def learn_about_LogisticRegression():

    # ── Learn About Logistic Regression ──────────────────────────────
    print("Logistic Regression is a supervised learning algorithm used for binary classification. Despite its name, it does not perform regression — it predicts the probability that an input belongs to class 1, then thresholds that probability into a class label (0 or 1).")
    print("\nKey Concepts:")
    print("1. Sigmoid Function: Instead of a raw linear output, logistic regression passes X·w+b through sigmoid: σ(z) = 1/(1+e⁻ᶻ). This squashes any value into (0, 1), giving us a probability.")
    print("2. Loss Function: Logistic regression uses Cross-Entropy loss (not MSE). It penalizes confident wrong predictions very heavily: J = -(1/m) Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]")
    print("3. Gradient Descent: Weights are updated the same way as linear regression — the cross-entropy gradient simplifies cleanly to dw = (1/m)·Xᵀ(ŷ−y).")
    print("4. Decision Boundary: predict() applies a threshold at 0.5 — if sigmoid output > 0.5 → class 1, else class 0.")
    print("\nApplications:")
    print("- Classifying emails as spam or not spam.")
    print("- Predicting whether a tumor is malignant or benign.")
    print("- Estimating the probability of customer churn.")

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.path import Path

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
    ax.text(0.5, 0.938, "LEARNML  |  How Logistic Regression works",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3)

    # ── Hyperparameter badges ─────────────────────────────────────────
    weights       = None
    n_iters       = 100000
    learning_rate = 0.01

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
            "idea": "Set all weights to zero and bias to zero. The model predicts 0.5 probability for everything yet.",
            "math": "self.weights = np.zeros(n_features)          self.bias = 0",
            "color": "#38BDF8", "edge": "#0284C7",
        },
        {
            "y": 0.618,
            "icon": "σ(z)",
            "title": "Sigmoid — Predict Probability",
            # y_predicted = sigmoid(np.dot(X, self.weights) + self.bias)
            # sigmoid squashes the linear output to a value between 0 and 1
            "idea": "Compute the linear score X·w+b then squash it through sigmoid. Output is now a probability.",
            "math": "y_pred = sigmoid( X · w + b )          σ(z) = 1 / (1 + e⁻ᶻ)",
            "color": "#A855F7", "edge": "#7E22CE",
        },
        {
            "y": 0.466,
            "icon": "J",
            "title": "Cross-Entropy Loss",
            # NOT MSE — cross-entropy penalizes confident wrong predictions heavily
            # J = -(1/n) * sum( y*log(y_pred) + (1-y)*log(1-y_pred) )
            "idea": "Cross-entropy loss — not MSE. It punishes confident wrong predictions exponentially harder.",
            "math": "J = -(1/m) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]",
            "color": "#EC4899", "edge": "#BE185D",
        },
        {
            "y": 0.314,
            "icon": "dJ",
            "title": "Compute Gradients",
            # dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            # db = (1/n_samples) * np.sum(y_predicted - y)
            # cross-entropy gradient simplifies to the same form as linear regression
            "idea": "Cross-entropy gradient simplifies cleanly. Same form as linear regression — just with sigmoid output.",
            "math": "dw = (1/m)·Xᵀ(ŷ−y)          db = (1/m)·Σ(ŷ−y)",
            "color": "#F59E0B", "edge": "#B45309",
        },
        {
            "y": 0.162,
            "icon": "0|1",
            "title": "Threshold → Class Label",
            # class_predicted = np.where(y_predicted > 0.5, 1, 0)
            # sigmoid returns probabilities — we threshold at 0.5 to get class labels
            "idea": "Sigmoid returns probabilities. Threshold at 0.5: above → class 1, below → class 0.",
            "math": "class = np.where(y_pred > 0.5,  1,  0)",
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
            ha="center", va="center", fontsize=8, fontweight="bold",
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
            ha="left", va="center", fontsize=9,
            color="#64748B", fontfamily="monospace", zorder=5)

    # ── Vertical arrows ───────────────────────────────────────────────
    for i in range(len(steps) - 1):
        y_start = steps[i]["y"]     - card_h / 2 - gap
        y_end   = steps[i + 1]["y"] + card_h / 2 + gap
        ax.annotate("", xy=(0.5, y_end), xytext=(0.5, y_start),
            arrowprops=dict(arrowstyle="-|>", color="#475569",
                            lw=1.8, mutation_scale=16), zorder=3)

    # ── Gradient descent loop arc (Step 4 → Step 2) ──────────────────
    # for _ in range(self.n_iters): sigmoid → loss → gradients → update weights
    s_start = steps[3]["y"]
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

    # ── Status bar ────────────────────────────────────────────────────
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


















def learn_about_NaiveBayes():

    # ── Learn About Gaussian Naive Bayes ────────────────────────────
    print(
        "Gaussian Naive Bayes is a supervised classification algorithm "
        "based on Bayes' theorem. It assumes that the features are "
        "conditionally independent given the class and models each "
        "feature using a Gaussian (normal) distribution."
    )

    print("\nKey Concepts:")
    print(
        "1. Classes: The model first finds all unique classes in the target y."
    )
    print(
        "2. Class Separation: For each class, the training samples belonging "
        "to that class are selected."
    )
    print(
        "3. Mean: For every feature, the model calculates the mean value "
        "for the current class."
    )
    print(
        "4. Variance: For every feature, the model calculates the variance "
        "for the current class."
    )
    print(
        "5. Prior: The model calculates how common each class is in the "
        "training data."
    )
    print(
        "6. Gaussian PDF: During prediction, the mean and variance are used "
        "to calculate the likelihood of each feature."
    )
    print(
        "7. Classification: The class with the highest posterior score "
        "is selected as the prediction."
    )

    print("\nApplications:")
    print("- Spam email classification")
    print("- Medical diagnosis classification")
    print("- Customer churn classification")
    print("- Document and text classification")


    # ── Figure Setup ─────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 13))

    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


    # ── Header ───────────────────────────────────────────────────────

    header = mpatches.FancyBboxPatch(
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
        "LEARNML  |  How Gaussian Naive Bayes works",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace",
        zorder=3
    )


    # ── Model information badges ────────────────────────────────────

    classes = "C = {0, 1, ...}"
    distribution = "Gaussian"
    assumption = "Conditional Independence"

    for i, (label, color) in enumerate([
        (classes, "#38BDF8"),
        (distribution, "#A855F7"),
        (assumption, "#10B981"),
    ]):

        bx = 0.08 + i * 0.305

        ax.add_patch(
            mpatches.FancyBboxPatch(
                (bx, 0.856), 0.285, 0.038,
                boxstyle="round,pad=0.010",
                facecolor="#1E293B",
                edgecolor=color,
                linewidth=1.2,
                zorder=2
            )
        )

        ax.text(
            bx + 0.1425,
            0.875,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontfamily="monospace",
            zorder=3
        )


    # ── Steps ────────────────────────────────────────────────────────
    #
    # These steps correspond directly to your fit() method.
    #
    # fit()
    #   │
    #   ├── find classes
    #   │
    #   └── for each class:
    #          X_c = X[y == c]
    #          mean
    #          variance
    #          prior
    #
    # The loop is repeated for every class.

    steps = [

        {
            "y": 0.770,
            "icon": "C",
            "title": "Find Classes",
            "idea": (
                "Find every unique class label in y. "
                "These are the possible classes the model can predict."
            ),
            "math": "_classes = np.unique(y)",
            "color": "#38BDF8",
            "edge": "#0284C7",
        },

        {
            "y": 0.618,
            "icon": "Xc",
            "title": "Separate by Class",
            "idea": (
                "Select only the training samples that belong to "
                "the current class."
            ),
            "math": "X_c = X[y == c]",
            "color": "#A855F7",
            "edge": "#7E22CE",
        },

        {
            "y": 0.466,
            "icon": "μ",
            "title": "Calculate Mean",
            "idea": (
                "For the current class, calculate the average of "
                "each feature."
            ),
            "math": "mean = X_c.mean(axis=0)",
            "color": "#EC4899",
            "edge": "#BE185D",
        },

        {
            "y": 0.314,
            "icon": "σ²",
            "title": "Calculate Variance",
            "idea": (
                "Measure how much each feature varies around "
                "its class mean."
            ),
            "math": "var = X_c.var(axis=0)",
            "color": "#F59E0B",
            "edge": "#B45309",
        },

        {
            "y": 0.162,
            "icon": "P",
            "title": "Calculate Prior",
            "idea": (
                "Calculate how frequently the current class appears "
                "in the training dataset."
            ),
            "math": "P(c) = n_c / n_samples",
            "color": "#FB923C",
            "edge": "#C2410C",
        },
    ]


    # ── Card dimensions ─────────────────────────────────────────────

    card_w = 0.68
    card_h = 0.112
    gap = 0.010
    icon_r = 0.038


    # ── Draw cards ──────────────────────────────────────────────────

    for step in steps:

        cy = step["y"]

        x0 = 0.5 - card_w / 2
        y0 = cy - card_h / 2


        # Shadow
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0 + 0.007, y0 - 0.007),
                card_w,
                card_h,
                boxstyle="round,pad=0.014",
                facecolor="#000000",
                edgecolor="none",
                alpha=0.35,
                zorder=2
            )
        )


        # Card
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y0),
                card_w,
                card_h,
                boxstyle="round,pad=0.014",
                facecolor="#1E293B",
                edgecolor=step["edge"],
                linewidth=1.8,
                zorder=3
            )
        )


        # Icon circle
        icon_x = x0 + 0.056

        ax.add_patch(
            plt.Circle(
                (icon_x, cy),
                icon_r,
                facecolor=step["edge"],
                zorder=4
            )
        )

        ax.text(
            icon_x,
            cy,
            step["icon"],
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#F8FAFC",
            fontfamily="monospace",
            zorder=5
        )


        # Title
        ax.text(
            x0 + 0.115,
            cy + 0.034,
            step["title"],
            ha="left",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=step["color"],
            fontfamily="sans-serif",
            zorder=5
        )


        # Plain English explanation
        ax.text(
            x0 + 0.115,
            cy + 0.006,
            step["idea"],
            ha="left",
            va="center",
            fontsize=8.8,
            color="#CBD5E1",
            fontfamily="sans-serif",
            zorder=5
        )


        # Divider
        ax.plot(
            [x0 + 0.115, x0 + card_w - 0.018],
            [cy - 0.018, cy - 0.018],
            color="#2D3748",
            lw=0.8,
            zorder=4
        )


        # Math
        ax.text(
            x0 + 0.115,
            cy - 0.036,
            step["math"],
            ha="left",
            va="center",
            fontsize=9.5,
            color="#64748B",
            fontfamily="monospace",
            zorder=5
        )


    # ── Vertical arrows ─────────────────────────────────────────────

    for i in range(len(steps) - 1):

        y_start = (
            steps[i]["y"]
            - card_h / 2
            - gap
        )

        y_end = (
            steps[i + 1]["y"]
            + card_h / 2
            + gap
        )

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


    # ── Class processing loop ───────────────────────────────────────
    #
    # The statistics are NOT calculated only once.
    #
    # They are calculated separately for every class:
    #
    # Class 0 → mean, variance, prior
    # Class 1 → mean, variance, prior
    # Class 2 → mean, variance, prior
    # ...
    #
    # until all classes have been processed.

    s_start = steps[4]["y"]
    s_mid = steps[2]["y"]
    s_end = steps[1]["y"]

    arc_x = 0.87

    verts = [
        (arc_x, s_start),
        (arc_x + 0.08, s_mid),
        (arc_x, s_end)
    ]

    codes = [
        Path.MOVETO,
        Path.CURVE3,
        Path.CURVE3
    ]

    ax.add_patch(
        mpatches.PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor="#F59E0B",
            linewidth=2,
            linestyle=(0, (5, 4)),
            zorder=3
        )
    )


    # Arrowhead pointing back toward "Separate by Class"

    ax.annotate(
        "",
        xy=(arc_x + 0.002, s_end),
        xytext=(arc_x + 0.022, s_end + 0.038),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#F59E0B",
            lw=1.5,
            mutation_scale=13
        ),
        zorder=4
    )


    # Loop label

    mid_y = (s_start + s_end) / 2

    ax.text(
        arc_x + 0.090,
        mid_y + 0.022,
        "Repeat for\neach class",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#F59E0B",
        fontfamily="sans-serif",
        zorder=5
    )

    ax.text(
        arc_x + 0.090,
        mid_y - 0.030,
        "class 0 → class 1\n→ ... → class n",
        ha="left",
        va="center",
        fontsize=8.2,
        color="#64748B",
        fontfamily="sans-serif",
        zorder=5
    )


    # ── Gaussian model information ─────────────────────────────────
    #
    # After training, every class has:
    #
    # mean
    # variance
    # prior
    #
    # These values are later used by _pdf() and _predict().

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.08, 0.022),
            0.84,
            0.052,
            boxstyle="round,pad=0.010",
            facecolor="#052e16",
            edgecolor="#059669",
            linewidth=1.2,
            zorder=2
        )
    )

    ax.text(
        0.5,
        0.048,
        "Model learned  —  mean μ  |  variance σ²  |  prior P(c)",
        ha="center",
        va="center",
        fontsize=10,
        color="#10B981",
        fontfamily="monospace",
        zorder=3
    )


    # ── Finish ──────────────────────────────────────────────────────

    plt.tight_layout()

    plt.show()
