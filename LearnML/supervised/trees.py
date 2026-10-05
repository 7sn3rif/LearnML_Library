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
