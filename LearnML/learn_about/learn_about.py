import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from matplotlib.path import Path





#learn about regression
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
 










#learn about classification

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



    

























def learn_about_DecisionTree():

    # ── Learn About Decision Trees ────────────────────────────────────
    print(
        "Decision Trees are a non-parametric supervised learning method used for both classification and regression tasks. "
        "The goal is to create a model that predicts the value of a target variable by learning simple decision rules "
        "inferred from the data features. A tree can be seen as a piecewise constant approximation."
    )
    print("\nKey Concepts:")
    print(
        "1. Model Representation: A tree consists of internal decision nodes (feature thresholds) and leaf nodes (final predictions). "
        "Samples traverse left if they satisfy the condition (e.g., feature <= threshold) and right otherwise."
    )
    print(
        "2. Impurity Criteria: The tree evaluates split quality using measures like Gini Impurity or Entropy for classification, "
        "or Variance Reduction for regression."
    )
    print(
        "3. Best Split Search: At each node, the algorithm greedily searches through all features and candidate thresholds "
        "to maximize Information Gain (impurity reduction)."
    )
    print(
        "4. Recursive Partitioning: The dataset is split into smaller subsets, and the process repeats recursively on each subset "
        "until stopping criteria (max depth, min samples) are triggered."
    )
    print("\nApplications:")
    print("- Medical diagnosis based on symptoms and patient metrics.")
    print("- Credit scoring and loan approval decisions.")
    print("- Customer churn prediction and feature importance analysis.")

    # ── Figure Setup (2 Columns) ───────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 13), gridspec_kw={'width_ratios': [1.15, 0.85]})
    fig.patch.set_facecolor("#0F172A")
    
    for ax in (ax1, ax2):
        ax.set_facecolor("#0F172A")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ==================================================================
    # LEFT PANEL: ALGORITHM PIPELINE FLOWCHART
    # ==================================================================

    # ── Header ────────────────────────────────────────────────────────
    header1 = mpatches.FancyBboxPatch(
        (0.08, 0.905), 0.84, 0.065,
        boxstyle="round,pad=0.01", facecolor="#1E293B",
        edgecolor="#38BDF8", linewidth=1.5, zorder=2
    )
    ax1.add_patch(header1)

    ax1.text(
        0.5, 0.938, "LEARNML  |  How Decision Trees Work",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3
    )

    # ── Hyperparameter badges ─────────────────────────────────────────
    root = None
    max_depth = 100
    criterion = "gini"

    fitted = root is not None
    crit_text = f"criterion = '{criterion}'"
    depth_text = f"max_depth = {max_depth}"
    st_text = "fitted \u2713" if fitted else "not fitted"
    st_color = "#10B981" if fitted else "#F59E0B"

    for i, (label, color) in enumerate([
        (crit_text, "#38BDF8"),
        (depth_text, "#A855F7"),
        (st_text, st_color),
    ]):
        bx = 0.06 + i * 0.31
        ax1.add_patch(
            mpatches.FancyBboxPatch(
                (bx, 0.856), 0.28, 0.038,
                boxstyle="round,pad=0.010", facecolor="#1E293B",
                edgecolor=color, linewidth=1.2, zorder=2
            )
        )
        ax1.text(
            bx + 0.14, 0.875, label,
            ha="center", va="center", fontsize=9,
            color=color, fontfamily="monospace", zorder=3
        )

    # ── Steps ─────────────────────────────────────────────────────────
    steps = [
        {
            "y": 0.770, "icon": "Stop?", "title": "Check Stopping Criteria",
            "idea": "If maximum depth is reached, sample size is too small, or node is pure, create a leaf.",
            "math": "n_labels == 1  |  depth >= max_depth  |  n < min_split",
            "color": "#38BDF8", "edge": "#0284C7",
        },
        {
            "y": 0.618, "icon": "f,t", "title": "Evaluate Split Candidates",
            "idea": "Extract unique values for every feature column to form candidate binary thresholds.",
            "math": "for f in features: thresholds = unique(X[:, f])",
            "color": "#A855F7", "edge": "#7E22CE",
        },
        {
            "y": 0.466, "icon": "Gain", "title": "Compute Information Gain",
            "idea": "Measure parent impurity minus weighted child impurities. Pick feature & threshold with max gain.",
            "math": "Gain = Impurity(y) - (N_l/N)*Imp(y_l) - (N_r/N)*Imp(y_r)",
            "color": "#EC4899", "edge": "#BE185D",
        },
        {
            "y": 0.314, "icon": "Split", "title": "Partition Dataset",
            "idea": "Divide current rows into left subset (X <= threshold) and right subset (X > threshold).",
            "math": "left_idxs = argwhere(X_col <= t)   right_idxs = argwhere(X_col > t)",
            "color": "#F59E0B", "edge": "#B45309",
        },
        {
            "y": 0.162, "icon": "Recur", "title": "Recurse Child Subtrees",
            "idea": "Call _grow_tree on left and right subsets with depth + 1, connecting child nodes to parent.",
            "math": "left = _grow_tree(X[left], depth+1)   right = ...",
            "color": "#FB923C", "edge": "#C2410C",
        },
    ]

    card_w, card_h = 0.68, 0.112
    gap, icon_r = 0.010, 0.038

    for step in steps:
        cy = step["y"]
        x0 = 0.44 - card_w / 2
        y0 = cy - card_h / 2

        # Shadow
        ax1.add_patch(
            mpatches.FancyBboxPatch(
                (x0 + 0.007, y0 - 0.007), card_w, card_h,
                boxstyle="round,pad=0.014", facecolor="#000000",
                edgecolor="none", alpha=0.35, zorder=2
            )
        )

        # Card
        ax1.add_patch(
            mpatches.FancyBboxPatch(
                (x0, y0), card_w, card_h,
                boxstyle="round,pad=0.014", facecolor="#1E293B",
                edgecolor=step["edge"], linewidth=1.8, zorder=3
            )
        )

        # Icon circle
        icon_x = x0 + 0.056
        ax1.add_patch(
            plt.Circle((icon_x, cy), icon_r, facecolor=step["edge"], zorder=4)
        )
        ax1.text(
            icon_x, cy, step["icon"],
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="#F8FAFC", fontfamily="monospace", zorder=5
        )

        # Title
        ax1.text(
            x0 + 0.115, cy + 0.034, step["title"],
            ha="left", va="center", fontsize=11.5, fontweight="bold",
            color=step["color"], fontfamily="sans-serif", zorder=5
        )

        # Plain English
        ax1.text(
            x0 + 0.115, cy + 0.006, step["idea"],
            ha="left", va="center", fontsize=8.5, color="#CBD5E1",
            fontfamily="sans-serif", zorder=5
        )

        # Divider
        ax1.plot(
            [x0 + 0.115, x0 + card_w - 0.018], [cy - 0.018, cy - 0.018],
            color="#2D3748", lw=0.8, zorder=4
        )

        # Math
        ax1.text(
            x0 + 0.115, cy - 0.036, step["math"],
            ha="left", va="center", fontsize=9,
            color="#64748B", fontfamily="monospace", zorder=5
        )

    # ── Vertical arrows ───────────────────────────────────────────────
    for i in range(len(steps) - 1):
        y_start = steps[i]["y"] - card_h / 2 - gap
        y_end = steps[i + 1]["y"] + card_h / 2 + gap
        ax1.annotate(
            "", xy=(0.44, y_end), xytext=(0.44, y_start),
            arrowprops=dict(
                arrowstyle="-|>", color="#475569", lw=1.8, mutation_scale=16
            ),
            zorder=3
        )

    # ── Recursive loop arc (Step 5 → Step 1) ──────────────────────────
    s_start, s_mid, s_end = steps[4]["y"], steps[2]["y"], steps[0]["y"]
    arc_x = 0.82

    verts = [(arc_x, s_start), (arc_x + 0.08, s_mid), (arc_x, s_end)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    ax1.add_patch(
        mpatches.PathPatch(
            Path(verts, codes), facecolor="none", edgecolor="#F59E0B",
            linewidth=2, linestyle=(0, (5, 4)), zorder=3
        )
    )

    ax1.annotate(
        "", xy=(arc_x + 0.002, s_end), xytext=(arc_x + 0.022, s_end - 0.038),
        arrowprops=dict(
            arrowstyle="-|>", color="#F59E0B", lw=1.5, mutation_scale=13
        ),
        zorder=4
    )

    mid_y = (s_start + s_end) / 2
    ax1.text(
        arc_x + 0.085, mid_y + 0.022, "Recurse for\neach child",
        ha="left", va="center", fontsize=8.5, fontweight="bold",
        color="#F59E0B", fontfamily="sans-serif", zorder=5
    )
    ax1.text(
        arc_x + 0.085, mid_y - 0.030, "until leaf\nnode hit",
        ha="left", va="center", fontsize=8,
        color="#64748B", fontfamily="sans-serif", zorder=5
    )

    # ── Convergence note ──────────────────────────────────────────────
    status = (
        f"Fitted  —  tree root created"
        if fitted
        else "Call  model.fit(X, y)  to train the decision tree"
    )
    color = "#10B981" if fitted else "#F59E0B"
    edge = "#059669" if fitted else "#B45309"
    bg = "#052e16" if fitted else "#1c1008"

    ax1.add_patch(
        mpatches.FancyBboxPatch(
            (0.06, 0.022), 0.88, 0.052,
            boxstyle="round,pad=0.010", facecolor=bg,
            edgecolor=edge, linewidth=1.2, zorder=2
        )
    )
    ax1.text(
        0.5, 0.048, status,
        ha="center", va="center", fontsize=10,
        color=color, fontfamily="monospace", zorder=3
    )

    # ==================================================================
    # RIGHT PANEL: VISUAL DECISION TREE GRAPH (YES / NO)
    # ==================================================================

    header2 = mpatches.FancyBboxPatch(
        (0.08, 0.905), 0.84, 0.065,
        boxstyle="round,pad=0.01", facecolor="#1E293B",
        edgecolor="#A855F7", linewidth=1.5, zorder=2
    )
    ax2.add_patch(header2)

    ax2.text(
        0.5, 0.938, "VISUAL STRUCTURE  |  Tree Predictions",
        ha="center", va="center", fontsize=12, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3
    )

    # Node rendering function
    def draw_node(x, y, text, is_leaf=False, is_green=False, w=0.30, h=0.065):
        box_color = "#059669" if (is_leaf and is_green) else ("#DC2626" if is_leaf else "#334155")
        border_color = "#10B981" if (is_leaf and is_green) else ("#EF4444" if is_leaf else "#38BDF8")
        text_color = "#F8FAFC"
        
        # Shadow
        ax2.add_patch(
            mpatches.FancyBboxPatch(
                (x - w / 2 + 0.005, y - h / 2 - 0.005), w, h,
                boxstyle="round,pad=0.010", facecolor="#000000",
                edgecolor="none", alpha=0.35, zorder=2
            )
        )
        # Box
        ax2.add_patch(
            mpatches.FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.010", facecolor=box_color,
                edgecolor=border_color, linewidth=1.8, zorder=3
            )
        )
        # Text
        ax2.text(
            x, y, text,
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=text_color, fontfamily="sans-serif", zorder=4
        )

    # Branch connection function with Yes / No label
    def draw_edge(p1, p2, label, label_color="#10B981"):
        ax2.annotate(
            "", xy=p2, xytext=p1,
            arrowprops=dict(
                arrowstyle="-|>", color="#94A3B8", lw=1.8, mutation_scale=14
            ),
            zorder=2
        )
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        # Text badge for Yes/No
        ax2.text(
            mx, my, f" {label} ",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=label_color, fontfamily="monospace", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor="none")
        )

    # Node positions
    node_root   = (0.50, 0.81)
    node_l1     = (0.26, 0.58)
    node_r1     = (0.74, 0.58)
    node_l2_l   = (0.13, 0.35)
    node_l2_r   = (0.39, 0.35)
    node_r2_l   = (0.61, 0.35)
    node_r2_r   = (0.87, 0.35)

    # Draw Nodes
    draw_node(node_root[0], node_root[1], "Age <= 30 ?", w=0.32)
    
    draw_node(node_l1[0], node_l1[1], "Income <= 50k ?", w=0.32)
    draw_node(node_r1[0], node_r1[1], "Student == Yes ?", w=0.32)

    draw_node(node_l2_l[0], node_l2_l[1], "Class: No ✗", is_leaf=True, is_green=False, w=0.22)
    draw_node(node_l2_r[0], node_l2_r[1], "Class: Yes ✓", is_leaf=True, is_green=True, w=0.22)
    draw_node(node_r2_l[0], node_r2_l[1], "Class: No ✗", is_leaf=True, is_green=False, w=0.22)
    draw_node(node_r2_r[0], node_r2_r[1], "Class: Yes ✓", is_leaf=True, is_green=True, w=0.22)

    # Draw Edges (Yes = Left, No = Right)
    h_offset = 0.033
    draw_edge((node_root[0] - 0.04, node_root[1] - h_offset), (node_l1[0] + 0.02, node_l1[1] + h_offset), "Yes", "#34D399")
    draw_edge((node_root[0] + 0.04, node_root[1] - h_offset), (node_r1[0] - 0.02, node_r1[1] + h_offset), "No", "#F87171")

    draw_edge((node_l1[0] - 0.03, node_l1[1] - h_offset), (node_l2_l[0] + 0.01, node_l2_l[1] + h_offset), "Yes", "#34D399")
    draw_edge((node_l1[0] + 0.03, node_l1[1] - h_offset), (node_l2_r[0] - 0.01, node_l2_r[1] + h_offset), "No", "#F87171")

    draw_edge((node_r1[0] - 0.03, node_r1[1] - h_offset), (node_r2_l[0] + 0.01, node_r2_l[1] + h_offset), "Yes", "#34D399")
    draw_edge((node_r1[0] + 0.03, node_r1[1] - h_offset), (node_r2_r[0] - 0.01, node_r2_r[1] + h_offset), "No", "#F87171")

    # Legend at bottom right
    ax2.add_patch(
        mpatches.FancyBboxPatch(
            (0.12, 0.12), 0.76, 0.12,
            boxstyle="round,pad=0.010", facecolor="#1E293B",
            edgecolor="#475569", linewidth=1.2, zorder=2
        )
    )
    ax2.text(
        0.5, 0.205, "TREE LEGEND",
        ha="center", va="center", fontsize=9.5, fontweight="bold",
        color="#CBD5E1", fontfamily="sans-serif", zorder=3
    )
    ax2.text(
        0.5, 0.155, "Decision Node: Evaluate feature condition\nGreen Branch (Yes): Condition is True\nRed Branch (No): Condition is False",
        ha="center", va="center", fontsize=8.5,
        color="#94A3B8", fontfamily="monospace", zorder=3
    )

    plt.tight_layout()
    plt.show()






















def learn_about_RandomForest():

    # ── Learn About Random Forest ─────────────────────────────────────
    print(
        "Random Forest is a supervised ensemble learning algorithm that combines "
        "multiple Decision Trees to make more accurate and robust predictions. "
        "Each tree is trained using a different bootstrap sample of the training data "
        "and considers random subsets of features when creating splits."
    )

    print("\nKey Concepts:")
    print(
        "1. Bootstrap Sampling: Each tree receives a random sample of the training "
        "data selected with replacement."
    )
    print(
        "2. Random Feature Selection: Each Decision Tree considers a random subset "
        "of features when searching for the best split."
    )
    print(
        "3. Multiple Decision Trees: Many independent Decision Trees are trained "
        "and stored inside the forest."
    )
    print(
        "4. Majority Voting: For classification, every tree predicts a class and "
        "the class with the most votes becomes the final prediction."
    )

    print("\nApplications:")
    print("- Customer churn prediction.")
    print("- Medical diagnosis.")
    print("- Fraud detection.")
    print("- Credit risk classification.")
    print("- General classification and regression tasks.")

    # ── Figure Setup (2 Columns) ──────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 13),
        gridspec_kw={'width_ratios': [1.15, 0.85]}
    )

    fig.patch.set_facecolor("#0F172A")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0F172A")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ==================================================================
    # LEFT PANEL: RANDOM FOREST ALGORITHM PIPELINE
    # ==================================================================

    # ── Header ────────────────────────────────────────────────────────
    header1 = mpatches.FancyBboxPatch(
        (0.08, 0.905), 0.84, 0.065,
        boxstyle="round,pad=0.01",
        facecolor="#1E293B",
        edgecolor="#38BDF8",
        linewidth=1.5,
        zorder=2
    )

    ax1.add_patch(header1)

    ax1.text(
        0.5, 0.938,
        "LEARNML  |  How Random Forest Works",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace",
        zorder=3
    )

    # ── Hyperparameter badges ─────────────────────────────────────────
    trees = 20
    max_depth = 10
    criterion = "gini"

    fitted = False

    trees_text = f"n_trees = {trees}"
    depth_text = f"max_depth = {max_depth}"
    crit_text = f"criterion = '{criterion}'"

    for i, (label, color) in enumerate([
        (trees_text, "#38BDF8"),
        (depth_text, "#A855F7"),
        (crit_text, "#10B981"),
    ]):

        bx = 0.06 + i * 0.31

        ax1.add_patch(
            mpatches.FancyBboxPatch(
                (bx, 0.856), 0.28, 0.038,
                boxstyle="round,pad=0.010",
                facecolor="#1E293B",
                edgecolor=color,
                linewidth=1.2,
                zorder=2
            )
        )

        ax1.text(
            bx + 0.14,
            0.875,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontfamily="monospace",
            zorder=3
        )

    # ── Steps ─────────────────────────────────────────────────────────
    steps = [

        {
            "y": 0.770,
            "icon": "X,y",
            "title": "Start With Training Data",
            "idea": "The forest receives the original training features X and target labels y.",
            "math": "X, y  →  Training Data",
            "color": "#38BDF8",
            "edge": "#0284C7",
        },

        {
            "y": 0.618,
            "icon": "Boot",
            "title": "Bootstrap Sampling",
            "idea": "Randomly sample training rows with replacement to create a dataset for each tree.",
            "math": "idxs = random.choice(n, n, replace=True)",
            "color": "#A855F7",
            "edge": "#7E22CE",
        },

        {
            "y": 0.466,
            "icon": "Tree",
            "title": "Train Decision Tree",
            "idea": "Build a Decision Tree using the bootstrap sample and random feature selection.",
            "math": "tree.fit(X_sample, y_sample)",
            "color": "#EC4899",
            "edge": "#BE185D",
        },

        {
            "y": 0.314,
            "icon": "Store",
            "title": "Store Tree in Forest",
            "idea": "Save the trained tree and repeat the process until n_trees is reached.",
            "math": "self.trees.append(tree)",
            "color": "#F59E0B",
            "edge": "#B45309",
        },

        {
            "y": 0.162,
            "icon": "Vote",
            "title": "Majority Voting",
            "idea": "Every tree predicts the class. The class with the most votes becomes the final prediction.",
            "math": "y_final = argmax(votes)",
            "color": "#FB923C",
            "edge": "#C2410C",
        },
    ]

    card_w = 0.68
    card_h = 0.112
    gap = 0.010
    icon_r = 0.038

    # ── Draw Cards ────────────────────────────────────────────────────
    for step in steps:

        cy = step["y"]

        x0 = 0.44 - card_w / 2
        y0 = cy - card_h / 2

        # Shadow
        ax1.add_patch(
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
        ax1.add_patch(
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

        # Icon
        icon_x = x0 + 0.056

        ax1.add_patch(
            plt.Circle(
                (icon_x, cy),
                icon_r,
                facecolor=step["edge"],
                zorder=4
            )
        )

        ax1.text(
            icon_x,
            cy,
            step["icon"],
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#F8FAFC",
            fontfamily="monospace",
            zorder=5
        )

        # Title
        ax1.text(
            x0 + 0.115,
            cy + 0.034,
            step["title"],
            ha="left",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=step["color"],
            fontfamily="sans-serif",
            zorder=5
        )

        # Explanation
        ax1.text(
            x0 + 0.115,
            cy + 0.006,
            step["idea"],
            ha="left",
            va="center",
            fontsize=8.5,
            color="#CBD5E1",
            fontfamily="sans-serif",
            zorder=5
        )

        # Divider
        ax1.plot(
            [x0 + 0.115, x0 + card_w - 0.018],
            [cy - 0.018, cy - 0.018],
            color="#2D3748",
            lw=0.8,
            zorder=4
        )

        # Math
        ax1.text(
            x0 + 0.115,
            cy - 0.036,
            step["math"],
            ha="left",
            va="center",
            fontsize=9,
            color="#64748B",
            fontfamily="monospace",
            zorder=5
        )

    # ── Vertical arrows ───────────────────────────────────────────────
    for i in range(len(steps) - 1):

        y_start = steps[i]["y"] - card_h / 2 - gap
        y_end = steps[i + 1]["y"] + card_h / 2 + gap

        ax1.annotate(
            "",
            xy=(0.44, y_end),
            xytext=(0.44, y_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#475569",
                lw=1.8,
                mutation_scale=16
            ),
            zorder=3
        )

    # ── Forest Loop: Step 4 → Step 2 ─────────────────────────────────
    s_start = steps[3]["y"]
    s_mid = steps[2]["y"]
    s_end = steps[1]["y"]

    arc_x = 0.82

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

    ax1.add_patch(
        mpatches.PathPatch(
            Path(verts, codes),
            facecolor="none",
            edgecolor="#F59E0B",
            linewidth=2,
            linestyle=(0, (5, 4)),
            zorder=3
        )
    )

    # Loop arrow
    ax1.annotate(
        "",
        xy=(arc_x + 0.002, s_end),
        xytext=(arc_x + 0.022, s_end - 0.038),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#F59E0B",
            lw=1.5,
            mutation_scale=13
        ),
        zorder=4
    )

    mid_y = (s_start + s_end) / 2

    ax1.text(
        arc_x + 0.085,
        mid_y + 0.022,
        "Build another\ntree",
        ha="left",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#F59E0B",
        fontfamily="sans-serif",
        zorder=5
    )

    ax1.text(
        arc_x + 0.085,
        mid_y - 0.030,
        f"until {trees}\ntrees",
        ha="left",
        va="center",
        fontsize=8,
        color="#64748B",
        fontfamily="sans-serif",
        zorder=5
    )

    # ── Status ────────────────────────────────────────────────────────
    status = (
        f"Fitted  —  {trees} Decision Trees in forest"
        if fitted
        else
        "Call  model.fit(X, y)  to train the Random Forest"
    )

    color = "#10B981" if fitted else "#F59E0B"
    edge = "#059669" if fitted else "#B45309"
    bg = "#052e16" if fitted else "#1c1008"

    ax1.add_patch(
        mpatches.FancyBboxPatch(
            (0.06, 0.022),
            0.88,
            0.052,
            boxstyle="round,pad=0.010",
            facecolor=bg,
            edgecolor=edge,
            linewidth=1.2,
            zorder=2
        )
    )

    ax1.text(
        0.5,
        0.048,
        status,
        ha="center",
        va="center",
        fontsize=10,
        color=color,
        fontfamily="monospace",
        zorder=3
    )

    # ==================================================================
    # RIGHT PANEL: RANDOM FOREST VISUAL STRUCTURE
    # ==================================================================

    header2 = mpatches.FancyBboxPatch(
        (0.08, 0.905),
        0.84,
        0.065,
        boxstyle="round,pad=0.01",
        facecolor="#1E293B",
        edgecolor="#A855F7",
        linewidth=1.5,
        zorder=2
    )

    ax2.add_patch(header2)

    ax2.text(
        0.5,
        0.938,
        "VISUAL STRUCTURE  |  Forest Voting",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace",
        zorder=3
    )

    # ── Tree rendering function ──────────────────────────────────────
    def draw_tree(x, y, title, prediction, color):

        # Tree box
        ax2.add_patch(
            mpatches.FancyBboxPatch(
                (x - 0.16, y - 0.075),
                0.32,
                0.15,
                boxstyle="round,pad=0.012",
                facecolor="#1E293B",
                edgecolor=color,
                linewidth=1.8,
                zorder=3
            )
        )

        # Tree title
        ax2.text(
            x,
            y + 0.040,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
            fontfamily="monospace",
            zorder=4
        )

        # Small tree representation
        ax2.text(
            x,
            y + 0.005,
            "Decision",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#CBD5E1",
            fontfamily="sans-serif",
            zorder=4
        )

        ax2.text(
            x,
            y - 0.020,
            "Tree",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#CBD5E1",
            fontfamily="sans-serif",
            zorder=4
        )

        # Prediction
        ax2.text(
            x,
            y - 0.052,
            f"Prediction → {prediction}",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#F8FAFC",
            fontfamily="monospace",
            zorder=4
        )

    # ── Three example trees ──────────────────────────────────────────
    draw_tree(
        0.28, 0.76,
        "TREE 1",
        "Class 1",
        "#38BDF8"
    )

    draw_tree(
        0.72, 0.76,
        "TREE 2",
        "Class 0",
        "#A855F7"
    )

    draw_tree(
        0.28, 0.51,
        "TREE 3",
        "Class 1",
        "#EC4899"
    )

    draw_tree(
        0.72, 0.51,
        "TREE ...",
        "Class 1",
        "#F59E0B"
    )

    # ── Arrows from trees to voting ──────────────────────────────────
    tree_points = [
        ((0.28, 0.685), (0.50, 0.36)),
        ((0.72, 0.685), (0.50, 0.36)),
        ((0.28, 0.435), (0.50, 0.36)),
        ((0.72, 0.435), (0.50, 0.36)),
    ]

    for p1, p2 in tree_points:

        ax2.annotate(
            "",
            xy=p2,
            xytext=p1,
            arrowprops=dict(
                arrowstyle="-|>",
                color="#64748B",
                lw=1.5,
                mutation_scale=13
            ),
            zorder=2
        )

    # ── Voting Box ───────────────────────────────────────────────────
    ax2.add_patch(
        mpatches.FancyBboxPatch(
            (0.30, 0.28),
            0.40,
            0.115,
            boxstyle="round,pad=0.012",
            facecolor="#1E293B",
            edgecolor="#F59E0B",
            linewidth=2,
            zorder=3
        )
    )

    ax2.text(
        0.50,
        0.365,
        "MAJORITY VOTING",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#F59E0B",
        fontfamily="monospace",
        zorder=4
    )

    ax2.text(
        0.50,
        0.325,
        "Class 1:  3 votes",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#CBD5E1",
        fontfamily="monospace",
        zorder=4
    )

    ax2.text(
        0.50,
        0.300,
        "Class 0:  1 vote",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#94A3B8",
        fontfamily="monospace",
        zorder=4
    )

    # ── Final Prediction Arrow ───────────────────────────────────────
    ax2.annotate(
        "",
        xy=(0.50, 0.205),
        xytext=(0.50, 0.28),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#10B981",
            lw=2,
            mutation_scale=14
        ),
        zorder=2
    )

    # ── Final Prediction ─────────────────────────────────────────────
    ax2.add_patch(
        mpatches.FancyBboxPatch(
            (0.25, 0.115),
            0.50,
            0.075,
            boxstyle="round,pad=0.012",
            facecolor="#052e16",
            edgecolor="#10B981",
            linewidth=1.8,
            zorder=3
        )
    )

    ax2.text(
        0.50,
        0.152,
        "FINAL PREDICTION",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#10B981",
        fontfamily="monospace",
        zorder=4
    )

    ax2.text(
        0.50,
        0.128,
        "Class 1  ✓",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace",
        zorder=4
    )

    # ── Legend ───────────────────────────────────────────────────────
    ax2.add_patch(
        mpatches.FancyBboxPatch(
            (0.12, 0.025),
            0.76,
            0.065,
            boxstyle="round,pad=0.010",
            facecolor="#1E293B",
            edgecolor="#475569",
            linewidth=1.2,
            zorder=2
        )
    )

    ax2.text(
        0.50,
        0.057,
        "Each tree votes independently → Forest selects the majority class",
        ha="center",
        va="center",
        fontsize=8,
        color="#94A3B8",
        fontfamily="monospace",
        zorder=3
    )

    plt.tight_layout()
    plt.show()
































import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def learn_about_SVM():

    # ── Educational explanation ───────────────────────────────────────

    print("=" * 65)
    print("LEARNML | SUPPORT VECTOR MACHINES")
    print("=" * 65)

    print("""
SVM is a supervised learning algorithm that finds a decision boundary
separating different classes while maximizing the margin.

IMPORTANT CONCEPTS
------------------
1. Decision Boundary:
   w · x + b = 0

2. Margin:
   The region between the boundaries w · x + b = +1 and -1.

3. Support Vectors:
   The training samples closest to the decision boundary.
   They help determine the optimal boundary.

4. Hinge Loss:
   Penalizes samples that are misclassified or inside the margin.

5. Kernel Trick:
   When data is not linearly separable, a kernel function measures
   similarity between samples in an implicit higher-dimensional space.

   Instead of explicitly transforming every sample, the kernel computes
   similarities directly.

COMMON KERNELS
--------------
Linear:       K(x, z) = x · z
Polynomial:   K(x, z) = (x · z + c)^d
RBF:          K(x, z) = exp(-gamma * ||x - z||²)

WHY USE A KERNEL?
-----------------
A linear SVM learns a straight decision boundary.
A kernel SVM can learn nonlinear boundaries, such as curved boundaries.

IMPORTANT IMPLEMENTATION NOTE
-----------------------------
The LinearSVM implementation uses a primal weight-vector approach.
Adding a kernel requires a different formulation, commonly the dual
optimization problem with an algorithm such as SMO.

The graph below is illustrative, not the output of a trained model.
""")

    # ── Figure setup ──────────────────────────────────────────────────

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(18, 11),
        gridspec_kw={"width_ratios": [1.08, 0.92]}
    )

    fig.patch.set_facecolor("#0F172A")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0F172A")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # ==================================================================
    # LEFT PANEL: SVM PIPELINE AND KERNEL TRICK
    # ==================================================================

    ax1.add_patch(
        FancyBboxPatch(
            (0.055, 0.915), 0.89, 0.06,
            boxstyle="round,pad=0.01",
            facecolor="#1E293B",
            edgecolor="#38BDF8",
            linewidth=1.5
        )
    )

    ax1.text(
        0.5, 0.945,
        "LEARNML | SVM TRAINING + KERNEL TRICK",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace"
    )

    # Badges
    badges = [
        ("Hinge Loss", "#38BDF8"),
        ("Maximum Margin", "#A855F7"),
        ("Nonlinear: Kernel", "#10B981")
    ]

    for i, (label, color) in enumerate(badges):
        x = 0.045 + i * 0.315

        ax1.add_patch(
            FancyBboxPatch(
                (x, 0.862), 0.295, 0.035,
                boxstyle="round,pad=0.007",
                facecolor="#1E293B",
                edgecolor=color,
                linewidth=1.1
            )
        )

        ax1.text(
            x + 0.1475, 0.879,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=color,
            fontfamily="monospace"
        )

    # The last card introduces the kernel trick as a separate concept.
    # It is not an additional operation in the existing LinearSVM class.

    steps = [
        {
            "y": 0.785,
            "title": "1. INPUT TRAINING DATA",
            "idea": "Features X and class labels y",
            "math": "X: (n_samples, n_features)",
            "color": "#38BDF8",
            "edge": "#0284C7"
        },
        {
            "y": 0.675,
            "title": "2. INITIALIZE PARAMETERS",
            "idea": "Initialize weights and bias",
            "math": "w = 0,  b = 0",
            "color": "#A855F7",
            "edge": "#7E22CE"
        },
        {
            "y": 0.565,
            "title": "3. CHECK THE MARGIN",
            "idea": "Convert labels to -1 and +1",
            "math": "yᵢ (w · xᵢ + b)",
            "color": "#EC4899",
            "edge": "#BE185D"
        },
        {
            "y": 0.455,
            "title": "4. UPDATE PARAMETERS",
            "idea": "Apply regularization and hinge-loss updates",
            "math": "If margin < 1: update w and b",
            "color": "#F59E0B",
            "edge": "#B45309"
        },
        {
            "y": 0.345,
            "title": "5. REPEAT TRAINING",
            "idea": "Process samples over multiple iterations",
            "math": "Repeat for n_iters",
            "color": "#FB923C",
            "edge": "#C2410C"
        },
        {
            "y": 0.235,
            "title": "6. LINEAR PREDICTION",
            "idea": "Calculate the score of a new sample",
            "math": "sign(w · x + b) → {-1, +1}",
            "color": "#10B981",
            "edge": "#059669"
        },
        {
            "y": 0.105,
            "title": "7. KERNEL TRICK: NONLINEAR DATA",
            "idea": "Use a kernel to compare samples implicitly",
            "math": "K(x, z) = similarity(x, z)",
            "color": "#F472B6",
            "edge": "#DB2777"
        }
    ]

    card_w = 0.84
    card_h = 0.078
    x0 = 0.08

    for step in steps:
        cy = step["y"]
        y0 = cy - card_h / 2

        # Shadow
        ax1.add_patch(
            FancyBboxPatch(
                (x0 + 0.006, y0 - 0.004),
                card_w, card_h,
                boxstyle="round,pad=0.01",
                facecolor="#000000",
                edgecolor="none",
                alpha=0.3
            )
        )

        # Card
        ax1.add_patch(
            FancyBboxPatch(
                (x0, y0),
                card_w, card_h,
                boxstyle="round,pad=0.01",
                facecolor="#1E293B",
                edgecolor=step["edge"],
                linewidth=1.4
            )
        )

        # Accent bar
        ax1.add_patch(
            FancyBboxPatch(
                (x0, y0),
                0.012, card_h,
                boxstyle="round,pad=0.001",
                facecolor=step["color"],
                edgecolor="none"
            )
        )

        ax1.text(
            0.5, cy + 0.021,
            step["title"],
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            color=step["color"]
        )

        ax1.text(
            0.5, cy - 0.001,
            step["idea"],
            ha="center",
            va="center",
            fontsize=7.9,
            color="#CBD5E1"
        )

        ax1.text(
            0.5, cy - 0.023,
            step["math"],
            ha="center",
            va="center",
            fontsize=8.0,
            color="#94A3B8",
            fontfamily="monospace"
        )

    # Arrows between the main training steps
    for i in range(len(steps) - 1):
        y_start = steps[i]["y"] - card_h / 2 - 0.003
        y_end = steps[i + 1]["y"] + card_h / 2 + 0.003

        ax1.annotate(
            "",
            xy=(0.5, y_end),
            xytext=(0.5, y_start),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#64748B",
                lw=1.3,
                mutation_scale=11
            )
        )

    # Highlight that the kernel trick is an alternative approach,
    # not a required final step in the linear SVM training loop.
    ax1.text(
        0.5, 0.035,
        "Linear SVM → straight boundary | Kernel SVM → nonlinear boundary",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#F472B6",
        fontfamily="monospace"
    )

    # ==================================================================
    # RIGHT PANEL: DECISION BOUNDARY AND EXACTLY TWO SUPPORT VECTORS
    # ==================================================================

    ax2.add_patch(
        FancyBboxPatch(
            (0.055, 0.915), 0.89, 0.06,
            boxstyle="round,pad=0.01",
            facecolor="#1E293B",
            edgecolor="#A855F7",
            linewidth=1.5
        )
    )

    ax2.text(
        0.5, 0.945,
        "VISUAL STRUCTURE | SUPPORT VECTORS",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace"
    )

    # Plot region
    plot_ax = fig.add_axes([0.565, 0.365, 0.385, 0.45])
    plot_ax.set_facecolor("#111827")

    for spine in plot_ax.spines.values():
        spine.set_color("#475569")

    plot_ax.tick_params(colors="#CBD5E1", labelsize=8)

    plot_ax.set_xlabel("Feature 1", color="#CBD5E1", fontsize=9)
    plot_ax.set_ylabel("Feature 2", color="#CBD5E1", fontsize=9)

   

    plot_ax.grid(
        True,
        color="#334155",
        alpha=0.45,
        linewidth=0.6
    )

    # Create two separated classes
    rng = np.random.default_rng(7)

    neg = rng.normal(
        loc=[-1.2, -0.8],
        scale=[0.35, 0.42],
        size=(22, 2)
    )

    pos = rng.normal(
        loc=[1.2, 0.8],
        scale=[0.35, 0.42],
        size=(22, 2)
    )

    # Two classes
    plot_ax.scatter(
        neg[:, 0],
        neg[:, 1],
        s=38,
        marker="o",
        c="#38BDF8",
        edgecolors="#E0F2FE",
        linewidths=0.5,
        label="Class -1",
        zorder=3
    )

    plot_ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=38,
        marker="^",
        c="#A855F7",
        edgecolors="#F3E8FF",
        linewidths=0.5,
        label="Class +1",
        zorder=3
    )

    # Linear decision boundary and margin lines
    xx = np.linspace(-2.4, 2.4, 300)

    boundary = -xx + 0.05
    margin_offset = 0.65

    plot_ax.plot(
        xx,
        boundary,
        color="#F8FAFC",
        linewidth=2.0,
        label="Decision boundary",
        zorder=4
    )

    plot_ax.plot(
        xx,
        boundary + margin_offset,
        color="#F59E0B",
        linewidth=1.5,
        linestyle="--",
        label="Margin",
        zorder=2
    )

    plot_ax.plot(
        xx,
        boundary - margin_offset,
        color="#F59E0B",
        linewidth=1.5,
        linestyle="--",
        zorder=2
    )

    # Exactly TWO illustrative support vectors: one from each class.
    # Their positions are selected to lie near opposite margins.
    support_vector_neg = np.array([-0.15, -0.30])
    support_vector_pos = np.array([0.15, 0.40])

    # Highlight the two support vectors
    plot_ax.scatter(
        support_vector_neg[0],
        support_vector_neg[1],
        s=190,
        marker="o",
        facecolors="none",
        edgecolors="#F59E0B",
        linewidths=2.2,
        zorder=6
    )

    plot_ax.scatter(
        support_vector_pos[0],
        support_vector_pos[1],
        s=190,
        marker="^",
        facecolors="none",
        edgecolors="#F59E0B",
        linewidths=2.2,
        zorder=6
    )

    # Connect each support vector to its corresponding class.
    # These are visual guides, not SVM optimization lines.
    neg_center = np.array([-1.2, -0.8])
    pos_center = np.array([1.2, 0.8])

    plot_ax.plot(
        [support_vector_neg[0], neg_center[0]],
        [support_vector_neg[1], neg_center[1]],
        color="#38BDF8",
        linestyle=":",
        linewidth=1.8,
        zorder=4
    )

    plot_ax.plot(
        [support_vector_pos[0], pos_center[0]],
        [support_vector_pos[1], pos_center[1]],
        color="#A855F7",
        linestyle=":",
        linewidth=1.8,
        zorder=4
    )

    # Labels for the two support vectors
    plot_ax.annotate(
        "SV: Class -1",
        xy=support_vector_neg,
        xytext=(-2.15, 0.05),
        color="#38BDF8",
        fontsize=8,
        arrowprops=dict(
            arrowstyle="->",
            color="#38BDF8",
            lw=1.2
        )
    )

    plot_ax.annotate(
        "SV: Class +1",
        xy=support_vector_pos,
        xytext=(0.65, 1.75),
        color="#C084FC",
        fontsize=8,
        arrowprops=dict(
            arrowstyle="->",
            color="#A855F7",
            lw=1.2
        )
    )

    plot_ax.set_xlim(-2.4, 2.4)
    plot_ax.set_ylim(-2.0, 2.4)

    legend = plot_ax.legend(
        loc="lower right",
        fontsize=7.2,
        facecolor="#1E293B",
        edgecolor="#475569",
        labelcolor="#F8FAFC",
        framealpha=0.95
    )

    for text_item in legend.get_texts():
        text_item.set_color("#F8FAFC")

    # ==================================================================
    # RIGHT PANEL: KERNEL TRICK EXPLANATION
    # ==================================================================

    kernel_cards = [
        (
            0.09, 0.245, 0.82, 0.075,
            "LINEAR SVM",
            "Straight decision boundary",
            "#38BDF8",
            "#0284C7"
        ),
        (
            0.09, 0.145, 0.82, 0.075,
            "KERNEL TRICK",
            "Implicit higher-dimensional feature space",
            "#F472B6",
            "#DB2777"
        ),
        (
            0.09, 0.045, 0.82, 0.075,
            "RBF KERNEL",
            "K(x,z) = exp(-gamma ||x-z||²)",
            "#A855F7",
            "#7E22CE"
        )
    ]

    for x, y, w, h, title, description, color, edge in kernel_cards:

        ax2.add_patch(
            FancyBboxPatch(
                (x, y),
                w, h,
                boxstyle="round,pad=0.012",
                facecolor="#1E293B",
                edgecolor=edge,
                linewidth=1.4
            )
        )

        ax2.text(
            x + w / 2,
            y + h * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color
        )

        ax2.text(
            x + w / 2,
            y + h * 0.30,
            description,
            ha="center",
            va="center",
            fontsize=8,
            color="#CBD5E1",
            fontfamily="monospace"
        )

    plt.show()




























#Neural Network Educational Explanation

def learn_about_optimizers():

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # ============================================================
    # LEARN ABOUT OPTIMIZERS
    # ============================================================

    print("""
============================================================
                 LEARN ABOUT OPTIMIZERS
============================================================

An optimizer is an algorithm responsible for updating the
weights and biases of a neural network during training.

The goal is to minimize the loss function.

TRAINING PROCESS:

1. Forward propagation:
   Calculate predictions using the current weights.

2. Loss calculation:
   Measure the difference between predictions and targets.

3. Backpropagation:
   Calculate gradients with respect to weights and biases.

4. Optimization:
   Use gradients to update the network parameters.

------------------------------------------------------------
1. GRADIENT DESCENT
------------------------------------------------------------

Gradient descent updates parameters in the opposite
direction of the gradient.

    W = W - learning_rate * dW
    b = b - learning_rate * db

Where:

W  = weights
b  = biases
dW = gradient of loss with respect to weights
db = gradient of loss with respect to biases

The learning rate controls the size of each update.

------------------------------------------------------------
2. STOCHASTIC GRADIENT DESCENT (SGD)
------------------------------------------------------------

    W = W - learning_rate * dW

SGD uses the calculated gradient to update parameters.

Advantages:
- Simple to implement.
- Low memory requirements.
- Easy to understand.

Disadvantages:
- Updates can fluctuate.
- Can converge slowly.
- Sensitive to the learning rate.

NOTE:
Your current NeuralNetwork.fit() uses the entire training
dataset for each iteration.

Therefore, your SGD implementation performs full-batch
gradient descent rather than stochastic or mini-batch SGD.

------------------------------------------------------------
3. ADAM
------------------------------------------------------------

Adam = Adaptive Moment Estimation.

It combines two ideas:

FIRST MOMENT:
An exponentially weighted average of past gradients.

    m = beta1 * m + (1 - beta1) * gradient

SECOND MOMENT:
An exponentially weighted average of squared gradients.

    v = beta2 * v + (1 - beta2) * gradient²

BIAS CORRECTION:

    m_corrected = m / (1 - beta1^t)
    v_corrected = v / (1 - beta2^t)

PARAMETER UPDATE:

    W = W - learning_rate *
        m_corrected / (sqrt(v_corrected) + epsilon)

Advantages:
- Adaptive parameter updates.
- Uses gradient history.
- Often converges quickly.

Disadvantages:
- Requires additional memory.
- Has more hyperparameters.
- Does not always generalize better than SGD.

------------------------------------------------------------
YOUR IMPLEMENTATION
------------------------------------------------------------

Your Layer.update() supports:

    optimizer = 'sgd'
    optimizer = 'adam'

Your NeuralNetwork._update() calls update() for every layer.

Your fit() method passes i + 1 as the Adam timestep.

============================================================
""")

    # ------------------------------------------------------------
    # FIGURE SETUP
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(17, 10), facecolor="#0F172A")

    fig.text(
        0.5, 0.96,
        "LEARNML | Understanding Optimizers",
        color="#38BDF8",
        fontsize=19,
        fontweight="bold",
        family="monospace",
        ha="center"
    )

    fig.text(
        0.5, 0.925,
        "How neural networks update weights and biases during training",
        color="#94A3B8",
        fontsize=10,
        family="monospace",
        ha="center"
    )

    # ------------------------------------------------------------
    # LEFT PANEL: OPTIMIZATION PIPELINE
    # ------------------------------------------------------------

    ax1 = fig.add_axes([0.04, 0.10, 0.43, 0.78])
    ax1.set_facecolor("#0F172A")
    ax1.axis("off")

    ax1.text(
        0.5, 0.96,
        "HOW OPTIMIZATION WORKS",
        color="#F8FAFC",
        fontsize=13,
        fontweight="bold",
        family="monospace",
        ha="center",
        transform=ax1.transAxes
    )

    # Each card represents one stage of training.
    stages = [
        (0.76, "LOSS FUNCTION",
         "Measures prediction error: L(y, y_pred)",
         "#F59E0B"),

        (0.56, "BACKPROPAGATION",
         "Calculates gradients: dW and db",
         "#A855F7"),

        (0.36, "OPTIMIZER",
         "Uses gradients to update parameters",
         "#38BDF8"),

        (0.16, "UPDATED PARAMETERS",
         "New weights and biases for next iteration",
         "#22C55E")
    ]

    for y, title, body, color in stages:

        # Draw a rounded card.
        card = FancyBboxPatch(
            (0.12, y), 0.76, 0.12,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#1E293B",
            transform=ax1.transAxes
        )

        ax1.add_patch(card)

        # Card heading.
        ax1.text(
            0.16, y + 0.085,
            title,
            color=color,
            fontsize=11,
            fontweight="bold",
            family="monospace",
            transform=ax1.transAxes
        )

        # Card description.
        ax1.text(
            0.16, y + 0.035,
            body,
            color="#F8FAFC",
            fontsize=8.5,
            family="monospace",
            transform=ax1.transAxes
        )

    # Connect the training stages with arrows.
    for y_start, y_end in [
        (0.76, 0.68),
        (0.56, 0.48),
        (0.36, 0.28)
    ]:

        arrow = FancyArrowPatch(
            (0.5, y_start),
            (0.5, y_end),
            arrowstyle="-|>",
            mutation_scale=15,
            color="#94A3B8",
            linewidth=1.8,
            transform=ax1.transAxes
        )

        ax1.add_patch(arrow)

    ax1.text(
        0.5, 0.06,
        "Repeat until the model learns",
        color="#22C55E",
        fontsize=10,
        family="monospace",
        ha="center",
        transform=ax1.transAxes
    )

    # ------------------------------------------------------------
    # RIGHT PANEL: SGD VS ADAM
    # ------------------------------------------------------------

    ax2 = fig.add_axes([0.51, 0.10, 0.45, 0.78])
    ax2.set_facecolor("#0F172A")
    ax2.axis("off")

    ax2.text(
        0.5, 0.96,
        "OPTIMIZER COMPARISON",
        color="#F8FAFC",
        fontsize=13,
        fontweight="bold",
        family="monospace",
        ha="center",
        transform=ax2.transAxes
    )

    # SGD card.
    card = FancyBboxPatch(
        (0.03, 0.68), 0.94, 0.21,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor="#38BDF8",
        facecolor="#1E293B",
        transform=ax2.transAxes
    )

    ax2.add_patch(card)

    ax2.text(
        0.07, 0.85,
        "SGD",
        color="#38BDF8",
        fontsize=12,
        fontweight="bold",
        family="monospace",
        transform=ax2.transAxes
    )

    ax2.text(
        0.07, 0.79,
        "W = W - learning_rate * dW",
        color="#F8FAFC",
        fontsize=10,
        family="monospace",
        transform=ax2.transAxes
    )

    ax2.text(
        0.07, 0.73,
        "Uses the current gradient directly.",
        color="#94A3B8",
        fontsize=9,
        family="monospace",
        transform=ax2.transAxes
    )

    # Adam card.
    card = FancyBboxPatch(
        (0.03, 0.40), 0.94, 0.23,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor="#A855F7",
        facecolor="#1E293B",
        transform=ax2.transAxes
    )

    ax2.add_patch(card)

    ax2.text(
        0.07, 0.59,
        "ADAM",
        color="#A855F7",
        fontsize=12,
        fontweight="bold",
        family="monospace",
        transform=ax2.transAxes
    )

    ax2.text(
        0.07, 0.535,
        "m = beta1 * m + (1-beta1) * dW",
        color="#F8FAFC",
        fontsize=9,
        family="monospace",
        transform=ax2.transAxes
    )

    ax2.text(
        0.07, 0.49,
        "v = beta2 * v + (1-beta2) * dW²",
        color="#F8FAFC",
        fontsize=9,
        family="monospace",
        transform=ax2.transAxes
    )

    ax2.text(
        0.07, 0.445,
        "Uses gradient history for adaptive updates.",
        color="#94A3B8",
        fontsize=8.5,
        family="monospace",
        transform=ax2.transAxes
    )

    # Comparison card.
    card = FancyBboxPatch(
        (0.03, 0.12), 0.94, 0.22,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor="#22C55E",
        facecolor="#1E293B",
        transform=ax2.transAxes
    )

    ax2.add_patch(card)

    ax2.text(
        0.07, 0.30,
        "WHEN SHOULD YOU USE THEM?",
        color="#22C55E",
        fontsize=11,
        fontweight="bold",
        family="monospace",
        transform=ax2.transAxes
    )

    ax2.text(
        0.07, 0.245,
        "SGD  -> Simple baseline and controlled experiments.",
        color="#F8FAFC",
        fontsize=8.5,
        family="monospace",
        transform=ax2.transAxes
    )

    ax2.text(
        0.07, 0.195,
        "ADAM -> Adaptive updates and a strong starting point.",
        color="#F8FAFC",
        fontsize=8.5,
        family="monospace",
        transform=ax2.transAxes
    )

    fig.text(
        0.5, 0.035,
        "LEARNML | OPTIMIZERS CONTROL HOW THE NETWORK LEARNS",
        color="#94A3B8",
        fontsize=9,
        family="monospace",
        ha="center"
    )

    plt.show()



def learn_about_activation_functions():

    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================================
    # LEARN ABOUT ACTIVATION FUNCTIONS
    # ============================================================

    print("""
============================================================
              LEARN ABOUT ACTIVATION FUNCTIONS
============================================================

An activation function transforms the output of a neuron.

A neuron first calculates:

    z = X @ W + b

Then applies an activation function:

    a = activation(z)

Where:

z = weighted sum before activation
a = output after activation

------------------------------------------------------------
WHY DO WE NEED ACTIVATION FUNCTIONS?
------------------------------------------------------------

Without nonlinear activation functions, stacking multiple
linear layers is still equivalent to one linear transformation.

Activation functions introduce nonlinearity, allowing
neural networks to learn complex patterns.

------------------------------------------------------------
1. LINEAR
------------------------------------------------------------

    f(x) = x

Returns the input without changing it.

Common use:
- Output layer for regression.

------------------------------------------------------------
2. RELU
------------------------------------------------------------

    f(x) = max(0, x)

Negative values become zero.
Positive values remain unchanged.

Common use:
- Hidden layers in many neural networks.

Advantages:
- Computationally simple.
- Helps reduce the vanishing gradient problem compared
  with sigmoid and tanh in many settings.

Disadvantage:
- Neurons can become inactive if they always output zero.

------------------------------------------------------------
3. SIGMOID
------------------------------------------------------------

    f(x) = 1 / (1 + e^(-x))

Maps values into the range (0, 1).

Common use:
- Binary classification output layer.
- Produces a probability-like score.

------------------------------------------------------------
4. TANH
------------------------------------------------------------

    f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

Maps values into the range (-1, 1).

Common use:
- Some hidden-layer architectures.
- Zero-centered activations.

------------------------------------------------------------
5. SOFTMAX
------------------------------------------------------------

    softmax(z_i) = exp(z_i) / sum(exp(z_j))

Converts a vector of scores into probabilities.

The probabilities sum to 1.

Common use:
- Multiclass classification output layer.

------------------------------------------------------------
YOUR IMPLEMENTATION
------------------------------------------------------------

Your Layer.forward() calculates:

    z = inputs @ weights + biases

Then selects an activation:

    relu
    sigmoid
    softmax
    linear

============================================================
""")

    # ------------------------------------------------------------
    # INPUT VALUES
    # ------------------------------------------------------------

    x = np.linspace(-5, 5, 500)

    # Calculate activation outputs.
    linear = x
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)

    # ------------------------------------------------------------
    # FIGURE SETUP
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(17, 11), facecolor="#0F172A")

    fig.text(
        0.5, 0.96,
        "LEARNML | Activation Functions",
        color="#38BDF8",
        fontsize=19,
        fontweight="bold",
        family="monospace",
        ha="center"
    )

    fig.text(
        0.5, 0.925,
        "How neurons transform weighted inputs into useful outputs",
        color="#94A3B8",
        fontsize=10,
        family="monospace",
        ha="center"
    )

    # ------------------------------------------------------------
    # GRAPH 1: LINEAR
    # ------------------------------------------------------------

    ax1 = fig.add_axes([0.06, 0.55, 0.40, 0.29])
    ax1.set_facecolor("#1E293B")

    ax1.plot(
        x, linear,
        color="#38BDF8",
        linewidth=2.8
    )

    ax1.axhline(0, color="#94A3B8", alpha=0.5)
    ax1.axvline(0, color="#94A3B8", alpha=0.5)

    ax1.set_title(
        "LINEAR",
        color="#38BDF8",
        fontsize=13,
        fontweight="bold",
        family="monospace"
    )

    ax1.text(
        0.5, 0.90,
        "f(x) = x",
        color="#F8FAFC",
        fontsize=10,
        family="monospace",
        ha="center",
        transform=ax1.transAxes
    )

    ax1.set_xlabel("Input (x)", color="#94A3B8")
    ax1.set_ylabel("Output", color="#94A3B8")

    ax1.tick_params(colors="#94A3B8")
    ax1.grid(color="#334155", linestyle="--", alpha=0.5)

    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # ------------------------------------------------------------
    # GRAPH 2: RELU
    # ------------------------------------------------------------

    ax2 = fig.add_axes([0.54, 0.55, 0.40, 0.29])
    ax2.set_facecolor("#1E293B")

    ax2.plot(
        x, relu,
        color="#22C55E",
        linewidth=2.8
    )

    ax2.axhline(0, color="#94A3B8", alpha=0.5)
    ax2.axvline(0, color="#94A3B8", alpha=0.5)

    ax2.set_title(
        "RELU",
        color="#22C55E",
        fontsize=13,
        fontweight="bold",
        family="monospace"
    )

    ax2.text(
        0.5, 0.90,
        "f(x) = max(0, x)",
        color="#F8FAFC",
        fontsize=10,
        family="monospace",
        ha="center",
        transform=ax2.transAxes
    )

    ax2.set_xlabel("Input (x)", color="#94A3B8")
    ax2.set_ylabel("Output", color="#94A3B8")

    ax2.tick_params(colors="#94A3B8")
    ax2.grid(color="#334155", linestyle="--", alpha=0.5)

    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # ------------------------------------------------------------
    # GRAPH 3: SIGMOID
    # ------------------------------------------------------------

    ax3 = fig.add_axes([0.06, 0.19, 0.40, 0.29])
    ax3.set_facecolor("#1E293B")

    ax3.plot(
        x, sigmoid,
        color="#F59E0B",
        linewidth=2.8
    )

    ax3.axhline(0, color="#94A3B8", alpha=0.5)
    ax3.axvline(0, color="#94A3B8", alpha=0.5)

    ax3.set_title(
        "SIGMOID",
        color="#F59E0B",
        fontsize=13,
        fontweight="bold",
        family="monospace"
    )

    ax3.text(
        0.5, 0.90,
        "f(x) = 1 / (1 + e^(-x))",
        color="#F8FAFC",
        fontsize=10,
        family="monospace",
        ha="center",
        transform=ax3.transAxes
    )

    ax3.set_xlabel("Input (x)", color="#94A3B8")
    ax3.set_ylabel("Output", color="#94A3B8")

    ax3.set_ylim(-0.1, 1.1)

    ax3.tick_params(colors="#94A3B8")
    ax3.grid(color="#334155", linestyle="--", alpha=0.5)

    for spine in ax3.spines.values():
        spine.set_color("#334155")

    # ------------------------------------------------------------
    # GRAPH 4: TANH
    # ------------------------------------------------------------

    ax4 = fig.add_axes([0.54, 0.19, 0.40, 0.29])
    ax4.set_facecolor("#1E293B")

    ax4.plot(
        x, tanh,
        color="#A855F7",
        linewidth=2.8
    )

    ax4.axhline(0, color="#94A3B8", alpha=0.5)
    ax4.axvline(0, color="#94A3B8", alpha=0.5)

    ax4.set_title(
        "TANH",
        color="#A855F7",
        fontsize=13,
        fontweight="bold",
        family="monospace"
    )

    ax4.text(
        0.5, 0.90,
        "f(x) = tanh(x)",
        color="#F8FAFC",
        fontsize=10,
        family="monospace",
        ha="center",
        transform=ax4.transAxes
    )

    ax4.set_xlabel("Input (x)", color="#94A3B8")
    ax4.set_ylabel("Output", color="#94A3B8")

    ax4.set_ylim(-1.2, 1.2)

    ax4.tick_params(colors="#94A3B8")
    ax4.grid(color="#334155", linestyle="--", alpha=0.5)

    for spine in ax4.spines.values():
        spine.set_color("#334155")

    # ------------------------------------------------------------
    # SOFTMAX EXAMPLE
    # ------------------------------------------------------------

    # Example scores produced by the output layer.
    logits = np.array([2.0, 1.0, 0.1])

    # Subtract maximum value for numerical stability.
    exp_values = np.exp(logits - np.max(logits))

    # Convert scores into probabilities.
    probabilities = exp_values / np.sum(exp_values)

    ax5 = fig.add_axes([0.08, 0.045, 0.84, 0.085])
    ax5.set_facecolor("#1E293B")

    classes = ["Class A", "Class B", "Class C"]

    bars = ax5.bar(
        classes,
        probabilities,
        color=["#38BDF8", "#A855F7", "#F59E0B"],
        width=0.5
    )

    ax5.set_ylim(0, 1.2)

    ax5.set_title(
        "SOFTMAX | Converts scores into probabilities",
        color="#F8FAFC",
        fontsize=10,
        family="monospace"
    )

    ax5.tick_params(colors="#94A3B8")

    for spine in ax5.spines.values():
        spine.set_color("#334155")

    # Display each predicted probability.
    for bar, probability in zip(bars, probabilities):

        ax5.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{probability:.2f}",
            color="#F8FAFC",
            fontsize=9,
            ha="center"
        )

    fig.text(
        0.5, 0.012,
        "SOFTMAX EXAMPLE | PROBABILITIES SUM TO 1.0",
        color="#38BDF8",
        fontsize=9,
        family="monospace",
        ha="center"
    )

    plt.show()



def learn_about_nn_loss_functions():

    import numpy as np
    import matplotlib.pyplot as plt

    # ============================================================
    # LEARN ABOUT NEURAL NETWORK LOSS FUNCTIONS
    # ============================================================

    print("""
============================================================
             LEARN ABOUT NN LOSS FUNCTIONS
============================================================

A loss function measures how far the network's predictions
are from the actual target values.

The goal of training is to minimize this loss.

------------------------------------------------------------
1. MEAN SQUARED ERROR (MSE)
------------------------------------------------------------

Used mainly for regression problems.

    MSE = (1/n) * sum((y_true - y_pred)²)

It calculates the average squared difference between
predictions and actual values.

Example:

    Actual:      [10, 20, 30]
    Predictions: [12, 18, 29]

    MSE = ((10-12)² + (20-18)² + (30-29)²) / 3

        = (4 + 4 + 1) / 3

        = 3

Large errors receive a larger penalty because the
differences are squared.

------------------------------------------------------------
2. BINARY CROSSENTROPY (BCE)
------------------------------------------------------------

Used for binary classification.

The target contains two possible classes:

    0 = Negative
    1 = Positive

Formula:

    BCE = -mean(
        y * log(p) + (1-y) * log(1-p)
    )

Where:

y = actual binary target
p = predicted probability

If the actual target is 1:
The loss penalizes low predicted probabilities.

If the actual target is 0:
The loss penalizes high predicted probabilities.

Typical output activation:
Sigmoid.

------------------------------------------------------------
3. CATEGORICAL CROSSENTROPY (CCE)
------------------------------------------------------------

Used for multiclass classification.

The target is represented using one-hot encoding.

Example:

    Class A = [1, 0, 0]
    Class B = [0, 1, 0]
    Class C = [0, 0, 1]

Formula:

    CCE = -(1/n) * sum(
        y_true * log(y_pred)
    )

For each sample, the loss focuses on the probability
assigned to the actual class.

Typical output activation:
Softmax.

------------------------------------------------------------
WHICH LOSS FUNCTION SHOULD YOU USE?
------------------------------------------------------------

Regression:
    Linear output + MSE

Binary classification:
    Sigmoid output + Binary Crossentropy

Multiclass classification:
    Softmax output + Categorical Crossentropy

------------------------------------------------------------
YOUR IMPLEMENTATION
------------------------------------------------------------

Your NeuralNetwork.compile() supports:

    model.compile(loss='mse')
    model.compile(loss='binary crossentropy')
    model.compile(loss='categorical crossentropy')

Your _compute_loss() returns:

    loss, dloss

loss:
The numerical error.

dloss:
The gradient of loss with respect to predictions.

The gradient is passed into _backward() to begin
backpropagation.

IMPORTANT:

For softmax + categorical crossentropy, the simplified
combined gradient is:

    (predictions - y_true) / batch_size

This assumes the softmax derivative is combined with
the categorical crossentropy derivative.

For sigmoid + binary crossentropy, your implementation
calculates the BCE derivative separately. Your sigmoid
layer's backward() method then applies its activation
derivative.

============================================================
""")

    # ------------------------------------------------------------
    # FIGURE SETUP
    # ------------------------------------------------------------

    fig = plt.figure(figsize=(17, 11), facecolor="#0F172A")

    fig.text(
        0.5, 0.96,
        "LEARNML | Neural Network Loss Functions",
        color="#38BDF8",
        fontsize=19,
        fontweight="bold",
        family="monospace",
        ha="center"
    )

    fig.text(
        0.5, 0.925,
        "How neural networks measure prediction errors",
        color="#94A3B8",
        fontsize=10,
        family="monospace",
        ha="center"
    )

    # ------------------------------------------------------------
    # GRAPH 1: MEAN SQUARED ERROR
    # ------------------------------------------------------------

    ax1 = fig.add_axes([0.06, 0.55, 0.40, 0.29])
    ax1.set_facecolor("#1E293B")

    # Assume the actual value is 3.
    actual = 3

    # Generate different predicted values.
    predictions = np.linspace(0, 6, 300)

    # Calculate squared errors.
    mse = (actual - predictions) ** 2

    ax1.plot(
        predictions,
        mse,
        color="#38BDF8",
        linewidth=2.8
    )

    # Mark the actual value.
    ax1.scatter(
        [actual],
        [0],
        color="#22C55E",
        s=90,
        zorder=5
    )

    ax1.axvline(
        actual,
        color="#22C55E",
        linestyle="--",
        alpha=0.7
    )

    ax1.set_title(
        "MEAN SQUARED ERROR (MSE)",
        color="#38BDF8",
        fontsize=12,
        fontweight="bold",
        family="monospace"
    )

    ax1.set_xlabel("Predicted value", color="#94A3B8")
    ax1.set_ylabel("Squared error", color="#94A3B8")

    ax1.tick_params(colors="#94A3B8")
    ax1.grid(color="#334155", linestyle="--", alpha=0.5)

    for spine in ax1.spines.values():
        spine.set_color("#334155")

    # ------------------------------------------------------------
    # GRAPH 2: BINARY CROSSENTROPY
    # ------------------------------------------------------------

    ax2 = fig.add_axes([0.54, 0.55, 0.40, 0.29])
    ax2.set_facecolor("#1E293B")

    # Predicted probabilities.
    p = np.linspace(0.001, 0.999, 400)

    # Loss when the actual class is 1.
    bce_positive = -np.log(p)

    # Loss when the actual class is 0.
    bce_negative = -np.log(1 - p)

    ax2.plot(
        p,
        bce_positive,
        color="#22C55E",
        linewidth=2.8,
        label="Actual class = 1"
    )

    ax2.plot(
        p,
        bce_negative,
        color="#F59E0B",
        linewidth=2.8,
        label="Actual class = 0"
    )

    ax2.set_title(
        "BINARY CROSSENTROPY (BCE)",
        color="#F59E0B",
        fontsize=12,
        fontweight="bold",
        family="monospace"
    )

    ax2.set_xlabel("Predicted probability", color="#94A3B8")
    ax2.set_ylabel("Loss", color="#94A3B8")

    ax2.tick_params(colors="#94A3B8")

    ax2.legend(
        facecolor="#1E293B",
        edgecolor="#334155",
        labelcolor="#F8FAFC",
        fontsize=8
    )

    ax2.grid(color="#334155", linestyle="--", alpha=0.5)

    for spine in ax2.spines.values():
        spine.set_color("#334155")

    # ------------------------------------------------------------
    # GRAPH 3: CATEGORICAL CROSSENTROPY
    # ------------------------------------------------------------

    ax3 = fig.add_axes([0.06, 0.15, 0.40, 0.29])
    ax3.set_facecolor("#1E293B")

    # Example predicted probabilities for three classes.
    class_names = ["Class A", "Class B", "Class C"]

    probabilities = np.array([0.70, 0.20, 0.10])

    # The actual class is Class A.
    actual_class = 0

    bars = ax3.bar(
        class_names,
        probabilities,
        color=["#22C55E", "#A855F7", "#F59E0B"],
        width=0.55
    )

    ax3.set_ylim(0, 1.15)

    ax3.set_title(
        "CATEGORICAL CROSSENTROPY (CCE)",
        color="#A855F7",
        fontsize=12,
        fontweight="bold",
        family="monospace"
    )

    ax3.set_ylabel("Predicted probability", color="#94A3B8")
    ax3.tick_params(colors="#94A3B8")

    for spine in ax3.spines.values():
        spine.set_color("#334155")

    # Display probabilities above the bars.
    for bar, probability in zip(bars, probabilities):

        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{probability:.2f}",
            color="#F8FAFC",
            fontsize=9,
            ha="center"
        )

    # Show the loss for the actual class.
    cce_loss = -np.log(probabilities[actual_class])

    ax3.text(
        0.5, -0.24,
        f"Actual: Class A | Loss = -log(0.70) = {cce_loss:.3f}",
        color="#22C55E",
        fontsize=8.5,
        family="monospace",
        ha="center",
        transform=ax3.transAxes
    )

    # ------------------------------------------------------------
    # RIGHT PANEL: WHEN TO USE EACH LOSS FUNCTION
    # ------------------------------------------------------------

    ax4 = fig.add_axes([0.54, 0.13, 0.40, 0.34])
    ax4.set_facecolor("#0F172A")
    ax4.axis("off")

    ax4.text(
        0.5, 0.98,
        "WHEN SHOULD YOU USE EACH LOSS?",
        color="#F8FAFC",
        fontsize=12,
        fontweight="bold",
        family="monospace",
        ha="center",
        transform=ax4.transAxes
    )

    # MSE card.
    card = FancyBboxPatch(
        (0.02, 0.68), 0.96, 0.23,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor="#38BDF8",
        facecolor="#1E293B",
        transform=ax4.transAxes
    )

    ax4.add_patch(card)

    ax4.text(
        0.07, 0.85,
        "MSE | REGRESSION",
        color="#38BDF8",
        fontsize=10,
        fontweight="bold",
        family="monospace",
        transform=ax4.transAxes
    )

    ax4.text(
        0.07, 0.77,
        "Linear output + MSE",
        color="#F8FAFC",
        fontsize=9,
        family="monospace",
        transform=ax4.transAxes
    )

    ax4.text(
        0.07, 0.715,
        "Predicting continuous numerical values.",
        color="#94A3B8",
        fontsize=8,
        family="monospace",
        transform=ax4.transAxes
    )

    # BCE card.
    card = FancyBboxPatch(
        (0.02, 0.37), 0.96, 0.23,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor="#F59E0B",
        facecolor="#1E293B",
        transform=ax4.transAxes
    )

    ax4.add_patch(card)

    ax4.text(
        0.07, 0.54,
        "BCE | BINARY CLASSIFICATION",
        color="#F59E0B",
        fontsize=10,
        fontweight="bold",
        family="monospace",
        transform=ax4.transAxes
    )

    ax4.text(
        0.07, 0.46,
        "Sigmoid output + BCE",
        color="#F8FAFC",
        fontsize=9,
        family="monospace",
        transform=ax4.transAxes
    )

    ax4.text(
        0.07, 0.405,
        "Predicting between two classes.",
        color="#94A3B8",
        fontsize=8,
        family="monospace",
        transform=ax4.transAxes
    )

    # CCE card.
    card = FancyBboxPatch(
        (0.02, 0.06), 0.96, 0.23,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.5,
        edgecolor="#A855F7",
        facecolor="#1E293B",
        transform=ax4.transAxes
    )

    ax4.add_patch(card)

    ax4.text(
        0.07, 0.23,
        "CCE | MULTICLASS CLASSIFICATION",
        color="#A855F7",
        fontsize=9.5,
        fontweight="bold",
        family="monospace",
        transform=ax4.transAxes
    )

    ax4.text(
        0.07, 0.15,
        "Softmax output + CCE",
        color="#F8FAFC",
        fontsize=9,
        family="monospace",
        transform=ax4.transAxes
    )

    ax4.text(
        0.07, 0.095,
        "Predicting among multiple classes.",
        color="#94A3B8",
        fontsize=8,
        family="monospace",
        transform=ax4.transAxes
    )

    fig.text(
        0.5, 0.035,
        "LEARNML | LOSS FUNCTIONS MEASURE HOW WRONG THE PREDICTIONS ARE",
        color="#94A3B8",
        fontsize=9,
        family="monospace",
        ha="center"
    )

    plt.show()





def learn_about_NeuralNetwork(model=None):

    # ==================================================================
    # EDUCATIONAL EXPLANATION
    # ==================================================================

    print("=" * 70)
    print("LEARNML | ARTIFICIAL NEURAL NETWORKS")
    print("=" * 70)

    print("""
A Neural Network is a supervised learning model inspired by the
structure of biological neural networks.

It consists of interconnected layers of neurons. Each neuron receives
inputs, multiplies them by weights, adds a bias, and applies an
activation function.

IMPORTANT CONCEPTS
------------------

1. Network Architecture:
   A neural network consists of an input layer, hidden layers,
   and an output layer.

2. Neurons:
   Each neuron calculates a weighted sum of its inputs and adds a bias.

   z = X @ W + b

   a = activation(z)

3. Weights and Biases:
   Weights determine the importance of connections between neurons.
   Biases allow neurons to shift their activation.

4. Activation Functions:
   Introduce nonlinearity into the network.

   ReLU:    f(x) = max(0, x)
   Sigmoid: f(x) = 1 / (1 + exp(-x))
   Softmax: Converts scores into probabilities.
   Linear:  f(x) = x

5. Forward Propagation:
   Input data travels from the input layer through hidden layers
   until the network produces predictions.

6. Loss Function:
   Measures the difference between predictions and actual targets.

   MSE:
       Mean Squared Error

   Binary Crossentropy:
       Used for binary classification.

   Categorical Crossentropy:
       Used for multiclass classification.

7. Backpropagation:
   Calculates gradients of the loss with respect to weights and
   biases using the chain rule.

   These gradients are propagated backward through the network.

8. Optimization:
   Updates weights and biases to reduce the loss.

   SGD:
       Uses the calculated gradients directly.

   Adam:
       Combines momentum and adaptive learning rates.

9. Training Iterations:
   Forward propagation, loss calculation, backpropagation,
   and parameter updates are repeated over multiple iterations.

IMPORTANT IMPLEMENTATION DETAILS
--------------------------------

This NeuralNetwork implementation uses NumPy to build the network
without relying on frameworks such as TensorFlow or PyTorch.

The fit() method performs full-batch training:

    predictions = self._forward(X)
    loss, dloss = self._compute_loss(predictions, y)
    self._backward(dloss)
    self._update(learning_rate, i + 1, optimizer)

Each iteration processes the entire training dataset and performs
one optimizer update.

The network uses separate Layer objects to store weights, biases,
activation functions, and gradients.

The graph below illustrates the architecture and training concepts.
""")

    # ==================================================================
    # DETERMINE NETWORK ARCHITECTURE
    # ==================================================================

    # If a model is supplied, inspect its actual Layer objects.
    # Otherwise, use an illustrative architecture.

    if model is not None and len(model.layers) > 0:

        layer_sizes = [model.layers[0].weights.shape[0]]

        layer_sizes.extend(
            layer.weights.shape[1]
            for layer in model.layers
        )

        activations = [
            getattr(layer, "activation", "unknown")
            for layer in model.layers
        ]

        loss_name = model.loss_name or "Not compiled"
        optimizer_name = model.optimizer_name or "Not compiled"

        architecture_source = "ACTUAL MODEL ARCHITECTURE"

    else:

        # Illustrative architecture:
        # 4 input features -> 5 -> 4 -> 3 output neurons

        layer_sizes = [4, 5, 4, 3]

        activations = [
            "ReLU",
            "ReLU",
            "Softmax"
        ]

        loss_name = "Categorical Crossentropy"
        optimizer_name = "Adam"

        architecture_source = "ILLUSTRATIVE ARCHITECTURE"

    # Layer labels
    layer_names = []

    for i in range(len(layer_sizes)):

        if i == 0:
            layer_names.append("INPUT")

        elif i == len(layer_sizes) - 1:
            layer_names.append("OUTPUT")

        else:
            layer_names.append(f"HIDDEN {i}")

    # Colors for different network layers
    layer_colors = [
        "#38BDF8",
        "#A855F7",
        "#EC4899",
        "#10B981",
        "#F59E0B",
        "#FB923C"
    ]

    # ==================================================================
    # FIGURE SETUP
    # ==================================================================

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(18, 11),
        gridspec_kw={"width_ratios": [1.15, 0.85]}
    )

    fig.patch.set_facecolor("#0F172A")

    for ax in (ax1, ax2):

        ax.set_facecolor("#0F172A")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.axis("off")

    # ==================================================================
    # LEFT PANEL: NEURAL NETWORK ARCHITECTURE
    # ==================================================================

    # ── Panel header ────────────────────────────────────────────────

    ax1.add_patch(
        FancyBboxPatch(
            (0.055, 0.915), 0.89, 0.06,
            boxstyle="round,pad=0.01",
            facecolor="#1E293B",
            edgecolor="#38BDF8",
            linewidth=1.5
        )
    )

    ax1.text(
        0.5, 0.945,
        "LEARNML | NEURAL NETWORK ARCHITECTURE",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace"
    )

    # ── Architecture information badge ──────────────────────────────

    ax1.add_patch(
        FancyBboxPatch(
            (0.12, 0.865), 0.76, 0.035,
            boxstyle="round,pad=0.007",
            facecolor="#1E293B",
            edgecolor="#A855F7",
            linewidth=1.1
        )
    )

    ax1.text(
        0.5, 0.882,
        architecture_source,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#A855F7",
        fontfamily="monospace"
    )

    # ── Network positions ───────────────────────────────────────────

    # Place each layer horizontally.
    # Neurons are distributed vertically within each layer.

    n_layers = len(layer_sizes)

    x_positions = np.linspace(
        0.12, 0.88, n_layers
    )

    # Store neuron coordinates for drawing connections later.
    neuron_positions = []

    # Display at most 8 neurons per layer to keep large networks readable.
    # The actual neuron count is still displayed below each layer.

    max_visible_neurons = 8

    for i, size in enumerate(layer_sizes):

        visible_count = min(size, max_visible_neurons)

        y_positions = np.linspace(
            0.72, 0.30, visible_count
        )

        positions = [
            (x_positions[i], y)
            for y in y_positions
        ]

        neuron_positions.append(positions)

    # ==================================================================
    # DRAW CONNECTIONS BETWEEN NEURONS
    # ==================================================================

    # Every neuron in one layer is connected to every neuron
    # in the next layer in a fully connected network.

    for layer_idx in range(n_layers - 1):

        current_layer = neuron_positions[layer_idx]
        next_layer = neuron_positions[layer_idx + 1]

        for x1, y1 in current_layer:

            for x2, y2 in next_layer:

                # Forward connections:
                # Represent weighted connections between neurons.

                connection = FancyArrowPatch(
                    (x1 + 0.018, y1),
                    (x2 - 0.018, y2),
                    arrowstyle="-",
                    color="#64748B",
                    linewidth=0.85,
                    alpha=0.42,
                    zorder=1
                )

                ax1.add_patch(connection)

    # ==================================================================
    # DRAW NEURONS
    # ==================================================================

    for i, positions in enumerate(neuron_positions):

        color = layer_colors[i % len(layer_colors)]

        for neuron_idx, (x, y) in enumerate(positions):

            # Neuron circle
            neuron = Circle(
                (x, y),
                radius=0.019,
                facecolor="#1E293B",
                edgecolor=color,
                linewidth=2,
                zorder=4
            )

            ax1.add_patch(neuron)

            # Small neuron identifier
            ax1.text(
                x, y,
                str(neuron_idx + 1),
                ha="center",
                va="center",
                fontsize=6.5,
                color="#F8FAFC",
                zorder=5
            )

        # Layer name
        ax1.text(
            x_positions[i],
            0.775,
            layer_names[i],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=color
        )

        # Actual number of neurons
        ax1.text(
            x_positions[i],
            0.255,
            f"{layer_sizes[i]} neurons",
            ha="center",
            va="center",
            fontsize=8,
            color="#CBD5E1",
            fontfamily="monospace"
        )

        # Activation function
        if i > 0:

            activation = str(activations[i - 1])

            ax1.text(
                x_positions[i],
                0.225,
                f"{activation}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="#94A3B8",
                fontfamily="monospace"
            )

        # Show omitted neurons if a layer has more than 8 neurons.
        if layer_sizes[i] > max_visible_neurons:

            ax1.text(
                x_positions[i],
                0.285,
                "...",
                ha="center",
                va="center",
                fontsize=12,
                color=color
            )

    # ==================================================================
    # FORWARD PROPAGATION DIRECTION
    # ==================================================================

    ax1.annotate(
        "",
        xy=(0.94, 0.825),
        xytext=(0.06, 0.825),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#38BDF8",
            lw=2,
            mutation_scale=15
        )
    )

    ax1.text(
        0.5, 0.845,
        "FORWARD PROPAGATION",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#38BDF8",
        fontfamily="monospace"
    )

    # ==================================================================
    # BACKPROPAGATION DIRECTION
    # ==================================================================

    ax1.annotate(
        "",
        xy=(0.06, 0.165),
        xytext=(0.94, 0.165),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#EC4899",
            lw=2,
            mutation_scale=15
        )
    )

    ax1.text(
        0.5, 0.14,
        "BACKPROPAGATION | GRADIENTS FLOW BACKWARD",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="#EC4899",
        fontfamily="monospace"
    )

    # ==================================================================
    # NETWORK FORMULA
    # ==================================================================

    ax1.add_patch(
        FancyBboxPatch(
            (0.10, 0.055), 0.80, 0.065,
            boxstyle="round,pad=0.01",
            facecolor="#1E293B",
            edgecolor="#38BDF8",
            linewidth=1.2
        )
    )

    ax1.text(
        0.5, 0.095,
        "z = X @ W + b",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#38BDF8",
        fontfamily="monospace"
    )

    ax1.text(
        0.5, 0.073,
        "a = activation(z)",
        ha="center",
        va="center",
        fontsize=9,
        color="#CBD5E1",
        fontfamily="monospace"
    )

    # ==================================================================
    # RIGHT PANEL: TRAINING MECHANICS
    # ==================================================================

    ax2.add_patch(
        FancyBboxPatch(
            (0.055, 0.915), 0.89, 0.06,
            boxstyle="round,pad=0.01",
            facecolor="#1E293B",
            edgecolor="#A855F7",
            linewidth=1.5
        )
    )

    ax2.text(
        0.5, 0.945,
        "TRAINING MECHANICS | HOW IT LEARNS",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#F8FAFC",
        fontfamily="monospace"
    )

    # ==================================================================
    # BADGES
    # ==================================================================

    badges = [
        ("FORWARD PASS", "#38BDF8"),
        ("BACKPROPAGATION", "#EC4899"),
        ("OPTIMIZATION", "#F59E0B")
    ]

    for i, (label, color) in enumerate(badges):

        x = 0.045 + i * 0.315

        ax2.add_patch(
            FancyBboxPatch(
                (x, 0.865), 0.295, 0.035,
                boxstyle="round,pad=0.007",
                facecolor="#1E293B",
                edgecolor=color,
                linewidth=1.1
            )
        )

        ax2.text(
            x + 0.1475, 0.882,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            fontfamily="monospace"
        )

    # ==================================================================
    # TRAINING CONCEPT CARDS
    # ==================================================================

    # These cards explain the mathematical operations used
    # by the NeuralNetwork class.

    training_cards = [

        {
            "x": 0.07,
            "y": 0.695,
            "w": 0.40,
            "h": 0.135,
            "title": "1. FORWARD PROPAGATION",
            "description": [
                "Pass input through every layer.",
                "Calculate weighted sums and",
                "apply activation functions."
            ],
            "formula": "a = f(X @ W + b)",
            "color": "#38BDF8",
            "edge": "#0284C7"
        },

        {
            "x": 0.53,
            "y": 0.695,
            "w": 0.40,
            "h": 0.135,
            "title": "2. LOSS CALCULATION",
            "description": [
                "Compare predictions with targets.",
                "Measure how wrong the network is.",
                "Calculate the loss gradient."
            ],
            "formula": "loss, dloss = _compute_loss()",
            "color": "#A855F7",
            "edge": "#7E22CE"
        },

        {
            "x": 0.07,
            "y": 0.525,
            "w": 0.40,
            "h": 0.135,
            "title": "3. BACKPROPAGATION",
            "description": [
                "Propagate gradients backward.",
                "Apply the chain rule to calculate",
                "weight and bias gradients."
            ],
            "formula": "self._backward(dloss)",
            "color": "#EC4899",
            "edge": "#BE185D"
        },

        {
            "x": 0.53,
            "y": 0.525,
            "w": 0.40,
            "h": 0.135,
            "title": "4. PARAMETER UPDATE",
            "description": [
                "Use calculated gradients to update",
                "weights and biases.",
                "Reduce the loss over time."
            ],
            "formula": "self._update(learning_rate, t, ...)",
            "color": "#F59E0B",
            "edge": "#B45309"
        }

    ]

    # Draw training cards
    for card in training_cards:

        x = card["x"]
        y = card["y"]
        w = card["w"]
        h = card["h"]

        # Card shadow
        ax2.add_patch(
            FancyBboxPatch(
                (x + 0.005, y - 0.005),
                w, h,
                boxstyle="round,pad=0.01",
                facecolor="#000000",
                edgecolor="none",
                alpha=0.3
            )
        )

        # Card body
        ax2.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.01",
                facecolor="#1E293B",
                edgecolor=card["edge"],
                linewidth=1.4
            )
        )

        # Colored accent
        ax2.add_patch(
            FancyBboxPatch(
                (x, y),
                0.009, h,
                boxstyle="round,pad=0.001",
                facecolor=card["color"],
                edgecolor="none"
            )
        )

        # Card title
        ax2.text(
            x + w / 2,
            y + h - 0.025,
            card["title"],
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=card["color"]
        )

        # Card descriptions
        for j, line in enumerate(card["description"]):

            ax2.text(
                x + w / 2,
                y + h - 0.052 - j * 0.019,
                line,
                ha="center",
                va="center",
                fontsize=7.4,
                color="#CBD5E1"
            )

        # Formula
        ax2.text(
            x + w / 2,
            y + 0.014,
            card["formula"],
            ha="center",
            va="center",
            fontsize=7,
            color="#94A3B8",
            fontfamily="monospace"
        )

    # ==================================================================
    # LOSS FUNCTIONS
    # ==================================================================

    ax2.add_patch(
        FancyBboxPatch(
            (0.07, 0.355), 0.86, 0.12,
            boxstyle="round,pad=0.01",
            facecolor="#1E293B",
            edgecolor="#A855F7",
            linewidth=1.4
        )
    )

    ax2.text(
        0.5, 0.447,
        "SUPPORTED LOSS FUNCTIONS",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#A855F7"
    )

    loss_items = [
        ("MSE", "Regression"),
        ("Binary Crossentropy", "Binary Classification"),
        ("Categorical Crossentropy", "Multiclass Classification")
    ]

    for i, (loss, purpose) in enumerate(loss_items):

        y = 0.418 - i * 0.027

        ax2.text(
            0.13, y,
            loss,
            ha="left",
            va="center",
            fontsize=7.7,
            color="#F8FAFC",
            fontfamily="monospace"
        )

        ax2.text(
            0.87, y,
            purpose,
            ha="right",
            va="center",
            fontsize=7.5,
            color="#94A3B8"
        )

    # ==================================================================
    # OPTIMIZERS
    # ==================================================================

    optimizer_cards = [

        (
            0.07, 0.195, 0.40, 0.115,
            "SGD",
            "W = W - learning_rate * dW",
            "Direct gradient update",
            "#38BDF8",
            "#0284C7"
        ),

        (
            0.53, 0.195, 0.40, 0.115,
            "ADAM",
            "m = moving average of gradients",
            "Adaptive moment estimation",
            "#F59E0B",
            "#B45309"
        )

    ]

    for x, y, w, h, title, formula, description, color, edge in optimizer_cards:

        ax2.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.01",
                facecolor="#1E293B",
                edgecolor=edge,
                linewidth=1.4
            )
        )

        ax2.text(
            x + w / 2,
            y + h * 0.75,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color
        )

        ax2.text(
            x + w / 2,
            y + h * 0.45,
            formula,
            ha="center",
            va="center",
            fontsize=7.1,
            color="#CBD5E1",
            fontfamily="monospace"
        )

        ax2.text(
            x + w / 2,
            y + h * 0.18,
            description,
            ha="center",
            va="center",
            fontsize=7.3,
            color="#94A3B8"
        )

    # ==================================================================
    # TRAINING CONFIGURATION
    # ==================================================================

    ax2.add_patch(
        FancyBboxPatch(
            (0.07, 0.065), 0.86, 0.09,
            boxstyle="round,pad=0.01",
            facecolor="#1E293B",
            edgecolor="#10B981",
            linewidth=1.4
        )
    )

    ax2.text(
        0.5, 0.132,
        "TRAINING CONFIGURATION",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#10B981"
    )

    ax2.text(
        0.5, 0.103,
        f"Loss: {loss_name}  |  Optimizer: {optimizer_name}",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#F8FAFC",
        fontfamily="monospace"
    )

    ax2.text(
        0.5, 0.080,
        "Full-batch training | Repeat for iterations",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#94A3B8",
        fontfamily="monospace"
    )

    # ==================================================================
    # DISPLAY FIGURE
    # ==================================================================

    plt.tight_layout()

    plt.savefig(
        "neural_network_pipeline.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0F172A"
    )

    plt.show()













