import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import matplotlib.path as Path


class LinearSVM:
 def __init__(self, learning_rate=0.001, lambda_param=0.01,n_iters=1000):  
    self.learning_rate = learning_rate  
    self.lambda_param = lambda_param  
    self.n_iters = n_iters  
    self.w = None  
    self.b = None  

 def fit(self, X, y):
   n_samples
