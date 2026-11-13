import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
import matplotlib.patheffects as pe
from matplotlib.path import Path

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    # Returns 0 if z is negative, or z itself if z is positive
    return np.maximum(0, z)

def softmax(z):
    # Subtracting np.max(z) for numerical stability (prevents overflow from e^z)
   shifted = z - np.max(z, axis=1, keepdims=True)
   exp_values = np.exp(shifted)
   return exp_values / np.sum(exp_values, axis=1, keepdims=True)




#layer class represents a single layer in a neural network, encapsulating its weights, biases, activation function , and methods for forward and backward propagation
class Layer:

    def __init__(self, n_inputs, n_neurons, activation='linear'):
        # Weight Initialization
        # He initialization (scaled by sqrt(2 / n_inputs)) is used for ReLU to keep the variance 
        # of activations stable across layers and prevent vanishing/exploding gradients.
        # Xavier/Glorot initialization (scaled by sqrt(1 / n_inputs)) is used for other activations.
        if activation == 'relu':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        else:
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)

        self.biases = np.zeros(n_neurons)
        self.activation = activation

        # Adam optimizer parameters for weights
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)

        # Adam optimizer parameters for biases
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs

        # Linear combination: matrix multiplication of inputs and weights plus the bias vector
        z = np.matmul(inputs, self.weights) + self.biases

        # Apply the corresponding activation function
        if self.activation == 'relu':
            self.output = relu(z)

        elif self.activation == 'sigmoid':
            self.output = sigmoid(z)

        elif self.activation == 'softmax':
            self.output = softmax(z)
            
        else:
            self.output = z

        return self.output

    def backward(self, dvalues):
        # Compute gradient with respect to the activation function output (d_activated)
        if self.activation == 'relu':
            # Derivative of ReLU: 1 where output > 0, else 0
            d_activated = dvalues * (self.output > 0)

        elif self.activation == 'sigmoid':
            # Derivative of sigmoid: output * (1 - output)
            d_activated = dvalues * self.output * (1 - self.output)

        elif self.activation == 'softmax':
            # Assumes dvalues is already (output - y_true) from a combined
            # softmax + categorical cross-entropy loss step.
            d_activated = dvalues

        else:
            d_activated = dvalues

        batch_size = self.inputs.shape[0]

        # Gradient of weights
        # Transposed inputs dotted with activated gradients yields the weight gradients.
        # (n_inputs, batch_size) @ (batch_size, n_neurons) -> (n_inputs, n_neurons)
        self.dweights = self.inputs.T @ d_activated

        # Gradient of biases: sum gradients across the batch dimension (axis 0)
        self.dbiases = np.sum(d_activated, axis=0)

        # Gradient of inputs (dinputs)
        # Passed backward to the preceding layer.
        # (batch_size, n_neurons) @ (n_neurons, n_inputs) -> (batch_size, n_inputs)
        self.dinputs = d_activated @ self.weights.T

    def update(self, learning_rate, t, optimizer='adam',
               beta1=0.9, beta2=0.999, epsilon=1e-8):
        if optimizer == 'sgd':
            # Standard SGD update rule
            self.weights -= learning_rate * self.dweights
            self.biases -= learning_rate * self.dbiases
        elif optimizer == 'adam':
            # Update first moment (exponential moving average of gradients / momentum)
            self.m_weights = beta1 * self.m_weights + (1 - beta1) * self.dweights
            self.m_biases = beta1 * self.m_biases + (1 - beta1) * self.dbiases

            # Update second moment (exponential moving average of squared gradients / uncentered variance)
            self.v_weights = beta2 * self.v_weights + (1 - beta2) * (self.dweights ** 2)
            self.v_biases = beta2 * self.v_biases + (1 - beta2) * (self.dbiases ** 2)

            # Bias Correction in Adam
            # Because moving averages are initialized at zero, they are biased toward zero, 
            # especially in early training steps (t). Dividing by (1 - beta^t) corrects this.
            m_weights_corrected = self.m_weights / (1 - beta1 ** t)
            m_biases_corrected = self.m_biases / (1 - beta1 ** t)

            v_weights_corrected = self.v_weights / (1 - beta2 ** t)
            v_biases_corrected = self.v_biases / (1 - beta2 ** t)

            # Update weights and biases using corrected moments and adaptive learning rates
            self.weights -= learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + epsilon)
            self.biases -= learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + epsilon)

        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")











#neural network class encapsulates the entire architecture, training, and prediction processes of a neural network model
class NeuralNetwork:

    def __init__(self, layers=None):
        self.layers = []
        if layers is not None:
            for layer in layers:
                self.add_layer(layer)

        self.loss_name = None
        self.optimizer_name = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def compile(self, loss='mse', optimizer='adam'):
        self.loss_name = loss.lower()
        if self.loss_name not in ['mse', 'binary crossentropy', 'categorical crossentropy']:
            raise ValueError(f"Unknown loss: {self.loss_name}")
        self.optimizer_name = optimizer.lower()
        if optimizer.lower() not in ['sgd', 'adam']:
            raise ValueError(f"Unknown optimizer: {optimizer}")    

    def _compute_loss(self, predictions, y_true):
        if self.loss_name == 'mse':
            loss = np.mean((predictions - y_true) ** 2)
            dloss = 2 * (predictions - y_true) / predictions.size
        elif self.loss_name == 'binary crossentropy':
            eps = 1e-12
            # Clip predictions to prevent log(0), which results in NaN values
            p = np.clip(predictions, eps, 1 - eps)
            loss = -np.mean(
                y_true * np.log(p) +
                (1 - y_true) * np.log(1 - p)
            )
            # Gradient of binary cross-entropy with respect to predictions
            dloss = (p - y_true) / (
                p * (1 - p) * predictions.shape[0]
            )
        elif self.loss_name == 'categorical crossentropy':
            eps = 1e-12
            # Clip predictions to prevent log(0)
            p = np.clip(predictions, eps, 1 - eps)
            batch_size = predictions.shape[0]
            loss = -np.sum(y_true * np.log(p)) / batch_size
            # When combined with softmax, the derivative simplifies cleanly to (p - y_true) / batch_size
            dloss = (p - y_true) / batch_size
        else:
            raise ValueError(f"Unknown loss: {self.loss_name}")
        return loss, dloss

    def _forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def _backward(self, loss_grad):
        grad = loss_grad
        # Propagate gradients backward through layers in reverse order using the chain rule
        for layer in reversed(self.layers):
            layer.backward(grad)
            grad = layer.dinputs

    def _update(self, learning_rate, t, optimizer):
        for layer in self.layers:
            layer.update(learning_rate, t, optimizer)

    def fit(self, X, y, learning_rate=0.01, iterations=1000, verbose=True):
        if self.loss_name is None:
            raise ValueError("Compile the model before training.")

        loss_history = []
        bar_length = 30

        for i in range(iterations):
            predictions = self._forward(X)
            loss, dloss = self._compute_loss(predictions, y)
            self._backward(dloss)
            self._update(learning_rate, i + 1, optimizer=self.optimizer_name)
            loss_history.append(loss)

            if verbose:
                progress = (i + 1) / iterations
                filled = int(bar_length * progress)
                bar = "=" * filled + ">" + "." * (bar_length - filled - 1)
                sys.stdout.write(
                    f"\rEpoch {i+1}/{iterations} [{bar}] - loss: {loss:.4f}"
                )
                sys.stdout.flush()

        if verbose:
            print()

        # ============================================================
        # GRAPHICAL REPRESENTATION OF NEURAL NETWORK TRAINING
        # ============================================================

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.patch.set_facecolor('#0F172A')
        ax.set_facecolor('#0F172A')

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # ── Title ──────────────────────────────────────────────
        ax.text(
            0.5, 0.94,
            "LEARNML | Neural Network Training Pipeline",
            ha='center', va='center',
            fontsize=20, fontweight='bold',
            color='#38BDF8', family='monospace'
        )

        ax.text(
            0.5, 0.895,
            "How a neural network learns through forward and backward propagation",
            ha='center', va='center',
            fontsize=10, color='#94A3B8'
        )

        # ── Training information ───────────────────────────────
        info = [
            ("OPTIMIZER", self.optimizer_name.upper()),
            ("LOSS", self.loss_name.upper()),
            ("ITERATIONS", str(iterations)),
            ("LEARNING RATE", str(learning_rate))
        ]

        info_x = [0.17, 0.39, 0.61, 0.83]

        for x, (label, value) in zip(info_x, info):

            box = FancyBboxPatch(
                (x - 0.09, 0.79), 0.18, 0.065,
                boxstyle="round,pad=0.008,rounding_size=0.01",
                facecolor='#1E293B',
                edgecolor='#334155',
                linewidth=1.2
            )

            ax.add_patch(box)

            ax.text(
                x, 0.83, label,
                ha='center', va='center',
                fontsize=8, color='#94A3B8',
                family='monospace'
            )

            ax.text(
                x, 0.805, value,
                ha='center', va='center',
                fontsize=10, color='#38BDF8',
                fontweight='bold'
            )

        # ── Pipeline stages ────────────────────────────────────
        stages = [
            {
                "x": 0.04,
                "color": "#38BDF8",
                "number": "01",
                "title": "Forward Propagation",
                "description": [
                    "Input passes through",
                    "every network layer.",
                    "",
                    "a = activation(XW + b)"
                ]
            },
            {
                "x": 0.235,
                "color": "#A855F7",
                "number": "02",
                "title": "Compute Loss",
                "description": [
                    "Compare predictions",
                    "with actual targets.",
                    "",
                    "Loss = MSE(y_pred, y)"
                ]
            },
            {
                "x": 0.43,
                "color": "#F472B6",
                "number": "03",
                "title": "Backpropagation",
                "description": [
                    "Calculate gradients",
                    "using the chain rule.",
                    "",
                    "∂Loss / ∂W"
                ]
            },
            {
                "x": 0.625,
                "color": "#F59E0B",
                "number": "04",
                "title": "Update Parameters",
                "description": [
                    "Optimizer updates",
                    "weights and biases.",
                    "",
                    "W = W - lr × gradient"
                ]
            },
            {
                "x": 0.82,
                "color": "#22C55E",
                "number": "05",
                "title": "Record & Repeat",
                "description": [
                    "Save current loss.",
                    "Repeat for next epoch.",
                    "",
                    "loss_history.append(loss)"
                ]
            }
        ]

        box_width = 0.155
        box_height = 0.28
        box_y = 0.43

        for stage in stages:

            x = stage["x"]
            color = stage["color"]

            # Main card
            card = FancyBboxPatch(
                (x, box_y),
                box_width,
                box_height,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                facecolor='#1E293B',
                edgecolor=color,
                linewidth=1.8
            )

            ax.add_patch(card)

            # Number
            ax.text(
                x + box_width / 2,
                box_y + box_height - 0.045,
                stage["number"],
                ha='center', va='center',
                fontsize=15,
                fontweight='bold',
                color=color,
                family='monospace'
            )

            # Title
            ax.text(
                x + box_width / 2,
                box_y + box_height - 0.095,
                stage["title"],
                ha='center', va='center',
                fontsize=9,
                fontweight='bold',
                color='white',
                wrap=True
            )

            # Description
            ax.text(
                x + box_width / 2,
                box_y + 0.09,
                "\n".join(stage["description"]),
                ha='center', va='center',
                fontsize=8,
                color='#CBD5E1',
                linespacing=1.6,
                family='monospace'
            )

        # ── Arrows between stages ──────────────────────────────
        for i in range(len(stages) - 1):

            start_x = stages[i]["x"] + box_width + 0.004
            end_x = stages[i + 1]["x"] - 0.004

            arrow = FancyArrowPatch(
                (start_x, box_y + box_height / 2),
                (end_x, box_y + box_height / 2),
                arrowstyle='-|>',
                mutation_scale=15,
                linewidth=1.8,
                color='#64748B'
            )

            ax.add_patch(arrow)

        # ── Training loop ──────────────────────────────────────
        vertices = [
            (0.897, 0.42),
            (0.897, 0.30),
            (0.117, 0.30),
            (0.117, 0.42)
        ]

        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO
        ]

        loop_path = Path(vertices, codes)

        loop = PathPatch(
            loop_path,
            edgecolor='#64748B',
            facecolor='none',
            linewidth=1.8,
            linestyle='--'
        )

        ax.add_patch(loop)

        # Loop arrow
        arrow = FancyArrowPatch(
            (0.117, 0.30),
            (0.117, 0.42),
            arrowstyle='-|>',
            mutation_scale=15,
            linewidth=1.8,
            color='#64748B'
        )

        ax.add_patch(arrow)

        ax.text(
            0.5, 0.27,
            "REPEAT UNTIL ITERATIONS ARE COMPLETED",
            ha='center', va='center',
            fontsize=9,
            color='#94A3B8',
            fontweight='bold',
            family='monospace'
        )

        # ── Final result ───────────────────────────────────────
        result = FancyBboxPatch(
            (0.25, 0.10), 0.5, 0.09,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor='#1E293B',
            edgecolor='#22C55E',
            linewidth=1.5
        )

        ax.add_patch(result)

        ax.text(
            0.5, 0.155,
            "TRAINING COMPLETED",
            ha='center', va='center',
            fontsize=12,
            color='#22C55E',
            fontweight='bold',
            family='monospace'
        )

        ax.text(
            0.5, 0.12,
            "Return loss_history",
            ha='center', va='center',
            fontsize=9,
            color='#CBD5E1'
        )

        # ── Bottom status ──────────────────────────────────────
        ax.text(
            0.5, 0.045,
            "LEARNML | FORWARD → LOSS → BACKWARD → UPDATE → REPEAT",
            ha='center', va='center',
            fontsize=9,
            color='#38BDF8',
            family='monospace'
        )

        plt.savefig(
            "neural_network_training_pipeline.png",
            dpi=300,
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )

        plt.show()

        return loss_history

    def predict(self, X):
        return self._forward(X)

    def predict_classes(self, X):
        predictions = self._forward(X)
        # Threshold at 0.5 for binary classification, otherwise use argmax for multi-class probabilities
        if predictions.shape[1] == 1:
            return (predictions >= 0.5).astype(int).ravel()
        return np.argmax(predictions, axis=1)

     # =========================================================
    # VISUALIZE TRAINED NEURAL NETWORK
    # =========================================================

    def visualize(self):

        if len(self.layers) == 0:
            print("The network has no layers.")
            return

        # ---------------------------------------------------------
        # Network information
        # ---------------------------------------------------------

        total_layers = len(self.layers)
        layer_sizes = [self.layers[0].weights.shape[0]]

        for layer in self.layers:
            layer_sizes.append(layer.weights.shape[1])

        total_parameters = sum(
            layer.weights.size + layer.biases.size
            for layer in self.layers
        )

        # ---------------------------------------------------------
        # Activation function names
        # ---------------------------------------------------------

        activation_names = []

        for layer in self.layers:
            activation_names.append(layer.activation.upper())

        # ---------------------------------------------------------
        # Figure
        # ---------------------------------------------------------

        fig, ax = plt.subplots(figsize=(16, 9))

        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.axis("off")

        # =========================================================
        # HEADER
        # =========================================================

        header = FancyBboxPatch(
            (0.20, 0.93),
            0.60,
            0.055,
            boxstyle="round,pad=0.01",
            facecolor="#1E293B",
            edgecolor="#38BDF8",
            linewidth=1.5
        )

        ax.add_patch(header)

        ax.text(
            0.5, 0.957,
            "LEARNML  |  Trained Neural Network",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#F8FAFC",
            fontfamily="monospace"
        )

        # =========================================================
        # INFORMATION PANEL
        # =========================================================

        info = FancyBboxPatch(
            (0.15, 0.85),
            0.70,
            0.055,
            boxstyle="round,pad=0.008",
            facecolor="#1E293B",
            edgecolor="#475569",
            linewidth=1
        )

        ax.add_patch(info)

        architecture = " → ".join(map(str, layer_sizes))

        ax.text(
            0.5, 0.877,
            f"LAYERS  {total_layers}     "
            f"NEURONS  {sum(layer_sizes)}     "
            f"PARAMETERS  {total_parameters:,}",
            ha="center",
            va="center",
            fontsize=9,
            color="#94A3B8",
            fontfamily="monospace"
        )

        ax.text(
            0.5, 0.81,
            f"ARCHITECTURE: {architecture}",
            ha="center",
            va="center",
            fontsize=10,
            color="#38BDF8",
            fontweight="bold",
            fontfamily="monospace"
        )

        # =========================================================
        # NETWORK LAYOUT
        # =========================================================

        layer_x = np.linspace(0.10, 0.90, total_layers + 1)

        neuron_positions = []

        for i, size in enumerate(layer_sizes):

            y_positions = np.linspace(
                0.22,
                0.68,
                min(size, 8)
            )

            if size == 1:
                y_positions = np.array([0.45])

            neuron_positions.append(y_positions)

        # =========================================================
        # CONNECTIONS
        # =========================================================

        for i in range(total_layers):

            input_positions = neuron_positions[i]
            output_positions = neuron_positions[i + 1]

            for input_y in input_positions:

                for output_y in output_positions:

                    ax.plot(
                        [layer_x[i], layer_x[i + 1]],
                        [input_y, output_y],
                        color="#475569",
                        linewidth=0.8,
                        alpha=0.55,
                        zorder=1
                    )

        # =========================================================
        # DRAW NEURONS
        # =========================================================

        for i, size in enumerate(layer_sizes):

            x = layer_x[i]
            positions = neuron_positions[i]

            if i == 0:
                neuron_color = "#38BDF8"
                layer_type = "INPUT"
            elif i == total_layers:
                neuron_color = "#22C55E"
                layer_type = "OUTPUT"
            else:
                neuron_color = "#A855F7"
                layer_type = "HIDDEN"

            for j, y in enumerate(positions):

                neuron = plt.Circle(
                    (x, y),
                    0.018,
                    facecolor=neuron_color,
                    edgecolor="#F8FAFC",
                    linewidth=1.2,
                    zorder=3
                )

                ax.add_patch(neuron)

                ax.text(
                    x,
                    y,
                    str(j + 1),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="#0F172A",
                    fontweight="bold",
                    zorder=4
                )

            # -----------------------------------------------------
            # Layer labels
            # -----------------------------------------------------

            ax.text(
                x,
                0.72,
                f"LAYER {i}",
                ha="center",
                va="center",
                fontsize=10,
                color=neuron_color,
                fontweight="bold",
                fontfamily="monospace"
            )

            ax.text(
                x,
                0.16,
                f"{layer_type}",
                ha="center",
                va="center",
                fontsize=9,
                color=neuron_color,
                fontweight="bold",
                fontfamily="monospace"
            )

            ax.text(
                x,
                0.12,
                f"{size} neurons",
                ha="center",
                va="center",
                fontsize=8,
                color="#CBD5E1",
                fontfamily="monospace"
            )

            # -----------------------------------------------------
            # Activation functions
            # -----------------------------------------------------

            if i > 0:

                activation_box = FancyBboxPatch(
                    (x - 0.055, 0.035),
                    0.11,
                    0.045,
                    boxstyle="round,pad=0.005",
                    facecolor="#1E293B",
                    edgecolor="#A855F7",
                    linewidth=1
                )

                ax.add_patch(activation_box)

                ax.text(
                    x,
                    0.057,
                    activation_names[i - 1],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#C084FC",
                    fontweight="bold",
                    fontfamily="monospace"
                )

        # =========================================================
        # LEGEND
        # =========================================================

        legend_items = [
            ("Input Layer", "#38BDF8"),
            ("Hidden Layer", "#A855F7"),
            ("Output Layer", "#22C55E")
        ]

        legend_x = [0.30, 0.50, 0.70]

        for x, (label, color) in zip(legend_x, legend_items):

            ax.scatter(
                x - 0.065,
                0.77,
                s=75,
                color=color,
                edgecolors="#F8FAFC"
            )

            ax.text(
                x - 0.045,
                0.77,
                label,
                ha="left",
                va="center",
                fontsize=8,
                color="#CBD5E1"
            )

        # =========================================================
        # SAVE AND DISPLAY
        # =========================================================

        plt.savefig(
            "neural_network_actual.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="#0F172A"
        )

        plt.show()     

