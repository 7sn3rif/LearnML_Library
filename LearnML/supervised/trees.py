import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path
from matplotlib.patches import Arc

#methods to calculate the impurity of a node in adecision trees
def entropy(y):
    if len(y) == 0:
        return 0.0
    # Get class proportions (p_1, p_2, ..., p_k)
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Adding 1e-9 (epsilon) prevents log2(0) without affecting precision
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))


def gini(y):
    if len(y) == 0:
        return 0.0
    
    # Get class proportions (p_1, p_2, ..., p_k)
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    
    # Formula: 1 - sum(p_i^2)
    return 1.0 - np.sum(probabilities ** 2)



#helper class to represent a node in a decision tree
class Node:
  def __init__(self , feature=None , threshold=None, left=None, right=None,*, value=None):
    self.feature=feature
    self.threshold=threshold
    self.left=left
    self.right=right
    self.value=value
  def is_leaf_node(self):
    return self.value is not None
  



class DecisionTree:
  def __init__(self, min_samples_split=2, max_depth=100, n_features=None,crietaion='gini'):
    self.min_samples_split=min_samples_split
    self.max_depth=max_depth
    self.n_features=n_features
    self.root=None
    self.crietaion=crietaion

  def _compute_impurity(self,y):
    #to choose which method to claculate the info gain
    if self.crietaion=='gini':
      return gini(y)
    elif self.crietaion=='entropy':
      return entropy(y)

#helper method to split the dataset based on a feature and a threshold, returning the indices of the left and right child nodes.
  def _split(self, X_column , split_threshold):
    left_idxs=np.argwhere(X_column<=split_threshold).flatten()
    right_idxs=np.argwhere(X_column>split_threshold).flatten()
    return left_idxs,right_idxs
  


  def _compute_gain(self,y,X_column,threshold):
    #parent impurity
    parent_impurity=self._compute_impurity(y)
    #create childrens
    left_idxs,right_idxs=self._split(X_column,threshold)
    #check if list are empty
    if len(left_idxs)==0 or len(right_idxs)==0:
      return 0
    #calculate wieghted average impurity of childern
    n_samp=len(y)
    n_left,n_right=len(left_idxs),len(right_idxs)
    impurity_l , impurity_r = self._compute_impurity(y[left_idxs]),self._compute_impurity(y[right_idxs])
    child_impurity=(n_left/n_samp)*impurity_l+(n_right/n_samp)*impurity_r
    #calculate the information gain
    information_gain=parent_impurity-child_impurity
    return information_gain

#helper method to find the most common label in a node, which is used to assign a class label to leaf nodes in the decision tree.
  def _most_common_label(self,y):
    labels, counts = np.unique(y, return_counts=True)
    return labels[np.argmax(counts)]

#helper method to find the best feature and threshold to split the data at a given node, based on maximizing information gain.
  def _best_split(self,X ,y,feature_idxs):
    best_gain=-1
    split_threshold=None
    split_feature=None
    for feat_idx in feature_idxs:
      X_column=X[:,feat_idx]#extract every feature column
      thresholds=np.unique(X_column)

      for threshold in thresholds:
        gain=self._compute_gain(y,X_column,threshold)
        if gain>best_gain:
          best_gain=gain
          split_threshold=threshold
          split_feature=feat_idx
    return split_feature,split_threshold,best_gain


#helper method to recursively grow the decision tree by splitting the data at each node based on the best feature and threshold, until stopping criteria are met (e.g., maximum depth or minimum samples per split).
  def _grow_tree(self,X,y,depth=0):
   n_labels=len(np.unique(y))
   n_samples,n_feats=X.shape
   #check stop crietira
   if (n_labels==1 or n_samples<self.min_samples_split or depth>=self.max_depth):
    #return the most common lablel on the leaf
     leaf_value=self._most_common_label(y)
     return Node(value=leaf_value)
     
   feature_idxs=np.random.choice(n_feats,self.n_features,replace=False)
   #find the best splits
   best_feature,best_threshold,best_gain=self._best_split(X,y,feature_idxs)
   
   if best_gain <= 0:
     leaf_value=self._most_common_label(y)
     return Node(value=leaf_value)
     
   #create child nodes 
   left_idxs,right_idxs=self._split(X[:,best_feature],best_threshold)
   left=self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
   right=self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)
   return Node(best_feature,best_threshold,left,right)

  def fit(self, X, y):
    #read the number of features
    self.n_features=X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
    self.root=self._grow_tree(X,y)
    # Graphical representation of the Decision Tree model training
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
    ax.text(0.5, 0.938, "LEARNML  |  CART Decision Tree Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold",
        color="#F8FAFC", fontfamily="monospace", zorder=3)
    
    # ── Step definitions ─────────────────────────────────────────────────
    steps = [
        {"y": 0.775, "title": "1. Check Stopping Criteria",
         "sub": "depth >= max_depth  |  n_samples < min_split",
         "color": "#38BDF8", "edge": "#0284C7"},
        {"y": 0.615, "title": "2. Evaluate Split Candidates",
         "sub": "Iterate feature columns & unique thresholds",
         "color": "#A855F7", "edge": "#7E22CE"},
        {"y": 0.455, "title": "3. Compute Impurity & Gain",
         "sub": "Gain = Impurity(y) − Σ (n_i/N) · Impurity(y_i)",
         "color": "#EC4899", "edge": "#BE185D"},
        {"y": 0.295, "title": "4. Split Data & Recurse",
         "sub": "left_child = _grow_tree(X_left, y_left)",
         "color": "#F59E0B", "edge": "#B45309"},
        {"y": 0.115, "title": "5. Return Leaf Node",
         "sub": "Predict majority class: _most_common_label(y)",
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
    
    # ── Recursive loop arc (Step 4 → Step 1) ────────────────────────────
    
    # Dashed curved path via bezier control points
    loop_x = [0.81, 0.96, 0.81]
    loop_y = [steps[3]["y"], 0.535, steps[0]["y"]]
    
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
    ax.annotate("", xy=(0.812, steps[0]["y"]), xytext=(0.83, steps[0]["y"] + 0.03),
        arrowprops=dict(arrowstyle="-|>", color="#F59E0B",
                        lw=1.5, mutation_scale=14), zorder=4)
    
    # Loop label
    ax.text(0.965, 0.535, "Recursive\nLoop",
        ha="left", va="center", fontsize=9.5, fontweight="bold",
        color="#F59E0B", fontfamily="monospace", zorder=5)
    ax.text(0.965, 0.495, "until leaf\nnode hit",
        ha="left", va="center", fontsize=8.5,
        color="#64748B", fontfamily="monospace", zorder=5)
    
    plt.tight_layout()
    plt.savefig("decision_tree_pipeline.png", dpi=150, bbox_inches="tight",
                facecolor="#0F172A")
    plt.show()

#helper method to traverse the decision tree for a given input sample, returning the predicted value or class label based on the splits defined by the tree nodes.
  def _traverse_tree(self, x , node):
    if node.is_leaf_node():
      return node.value
    if x[node.feature]<=node.threshold:
      return self._traverse_tree(x,node.left)
    return self._traverse_tree(x,node.right)


  def predict (self , X):
    predictions = []
    for x in X:
        predictions.append(self._traverse_tree(x, self.root))
    return np.array(predictions)

  
  