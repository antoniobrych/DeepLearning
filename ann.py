import numpy as np
from typing import Callable, Literal, Optional
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

Activation = Callable[[np.ndarray], np.ndarray]
LossFn     = Callable[[np.ndarray, np.ndarray], float]

class ANN:
    """
    Feedforward net with flexible sizing (no explicit per-layer bias vectors).
    If you want a bias, prepend a column of ones to your inputs as you suggested.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        net_length: int,
        net_width: int | list[int],
        hidden_activation: Activation | Literal["relu","sigmoid","tanh","leaky_relu"],
        output_activation: Activation | Literal["identity","sigmoid","tanh","softmax"],
        loss: LossFn | Literal["mse","bce","cross_entropy"],
        seed: Optional[int] = None,
    ) -> None:
        self.hidden_activation_name = hidden_activation if isinstance(hidden_activation, str) else "custom"
        self.output_activation_name = output_activation if isinstance(output_activation, str) else "custom"
        self.loss_name = loss if isinstance(loss, str) else "custom"

        (self.hidden_activation, self.hidden_activation_deriv) = self._resolve_activation(self.hidden_activation_name, hidden_activation)
        (self.output_activation, self.output_activation_deriv) = self._resolve_activation(self.output_activation_name, output_activation)
        (self.loss_fn, self.loss_deriv) = self._resolve_loss(self.loss_name, loss)

        self.seed = seed
        self.input_size  = input_size
        self.output_size = output_size
        self.net_length  = net_length
        self.net_width   = net_width if isinstance(net_width, list) else [net_width] * net_length
        if not isinstance(self.net_width, list):
            raise ValueError("net_width must be a list of integers.")
        if not all(isinstance(i, int) for i in self.net_width):
            raise ValueError("All elements in net_width must be integers.")

    def _resolve_activation(self, name: str, fn: Activation | str):
        if callable(fn):
            raise NotImplementedError(
                "Derivatives for custom callable activations are not supported. "
                "Please use a string identifier or modify the class."
            )

        def relu(x): return np.maximum(0, x)
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
        def tanh(x): return np.tanh(x)
        def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
        def identity(x): return x
        def softmax(x):
            z = x - np.max(x, axis=1, keepdims=True)
            e_z = np.exp(z)
            return e_z / np.sum(e_z, axis=1, keepdims=True)

        def relu_deriv(x): return np.where(x > 0, 1.0, 0.0)
        def sigmoid_deriv(x):
            s = sigmoid(x)
            return s * (1 - s)
        def tanh_deriv(x): return 1 - np.tanh(x)**2
        def leaky_relu_deriv(x, a=0.01): return np.where(x > 0, 1.0, a)
        def identity_deriv(x): return np.ones_like(x)
        def softmax_deriv(x): return np.ones_like(x)  

        lut = {
            "relu":        (relu, relu_deriv),
            "sigmoid":     (sigmoid, sigmoid_deriv),
            "tanh":        (tanh, tanh_deriv),
            "leaky_relu":  (leaky_relu, leaky_relu_deriv),
            "identity":    (identity, identity_deriv),
            "softmax":     (softmax, softmax_deriv),
        }
        if name in lut:
            return lut[name]
        raise ValueError(f"Unknown activation: {name!r}")

    def _resolve_loss(self, name: str, fn: LossFn | str):
        if callable(fn):
            raise NotImplementedError(
                "Derivatives for custom callable losses are not supported."
            )

        def _mse(y, t):
            return np.mean((y - t) ** 2)

        def _bce(y, t, eps=1e-12):
            y = np.clip(y, eps, 1 - eps)
            return -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))

        def _cross_entropy(y, t, eps=1e-12):
            y = np.clip(y, eps, 1 - eps)
            if t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1):
                t = t.reshape(-1)
                return -np.mean(np.log(y[np.arange(y.shape[0]), t]))
            return -np.mean(np.sum(t * np.log(y), axis=1))

        def _mse_deriv(y, t): return y - t

        def _bce_deriv(y, t, eps=1e-12):
            y = np.clip(y, eps, 1 - eps)
            return (y - t) / (y * (1 - y))

        def _cross_entropy_deriv(y, t, eps=1e-12):
            y = np.clip(y, eps, 1 - eps)
            return -t / y

        lut = {
            "mse":           (_mse, _mse_deriv),
            "bce":           (_bce, _bce_deriv),
            "cross_entropy": (_cross_entropy, _cross_entropy_deriv),
        }
        if name in lut:
            return lut[name]
        raise ValueError(f"Unknown loss: {name!r}")

    def _initialize_weights(self, X: np.ndarray) -> None:
        self.weights = []
        rng = np.random.default_rng(self.seed)
        layer_sizes = [self.input_size] + self.net_width + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            in_size = layer_sizes[i]
            # He for ReLU hidden layers, otherwise Xavier-ish
            if self.hidden_activation_name == 'relu' and i < len(layer_sizes) - 2:
                std = np.sqrt(2.0 / in_size)
            else:
                std = np.sqrt(1.0 / in_size)
            self.weights.append(rng.normal(0, std, size=(in_size, layer_sizes[i+1])))

    def _forward_pass(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not hasattr(self, "weights") or not self.weights:
            self._initialize_weights(X)
        activations = [X]
        pre_activations = []
        for W in self.weights[:-1]:
            z = activations[-1] @ W
            a = self.hidden_activation(z)
            pre_activations.append(z)
            activations.append(a)
        z = activations[-1] @ self.weights[-1]
        a = self.output_activation(z)
        pre_activations.append(z)
        activations.append(a)
        return activations, pre_activations

    def _backward_pass(self, y_true, activations, pre_activations):
        if not activations or not pre_activations:
            raise ValueError("Run forward pass first.")
        L = len(self.weights)
        grads = [None] * L
        y_pred = activations[-1]
        N = y_true.shape[0]

        if self.output_activation_name == "softmax" and self.loss_name == "cross_entropy":
            if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
                y_idx = y_true.reshape(-1)
                y_t = np.zeros_like(y_pred)
                y_t[np.arange(N), y_idx] = 1.0
            else:
                y_t = y_true
            delta = (y_pred - y_t) / N
        else:
            d_loss_dy = self.loss_deriv(y_pred, y_true)
            d_y_dz = self.output_activation_deriv(pre_activations[-1])
            delta = (d_loss_dy * d_y_dz) / N

        grads[-1] = activations[-2].T @ delta
        for l in range(L-1, 0, -1):
            delta = (delta @ self.weights[l].T) * self.hidden_activation_deriv(pre_activations[l-1])
            grads[l-1] = activations[l-1].T @ delta
        return grads

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        lr: float = 0.1,
        epochs: int = 300,
        batch_size: Optional[int] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: int = 0,
    ):
        N = X.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size > N:
            batch_size = N 
        loss_hist, val_loss_hist = [], []

        for epoch in range(1, epochs + 1):
            idx = np.arange(N)
            np.random.shuffle(idx)
            for start in range(0, N, batch_size):
                b_idx = idx[start:start + batch_size]
                Xb, yb = X[b_idx], y[b_idx]
                activations, pre_activations = self._forward_pass(Xb)
                grads = self._backward_pass(yb, activations, pre_activations)
                for i in range(len(self.weights)):
                    self.weights[i] -= lr * grads[i]

            train_pred = self.predict_proba(X)
            train_loss = self.loss_fn(train_pred, y)
            loss_hist.append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self.loss_fn(val_pred, y_val)
                val_loss_hist.append(val_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
                if X_val is not None and y_val is not None:
                    print(f"epoch {epoch:4d} | loss {train_loss:.4f} | val {val_loss:.4f}")
                else:
                    print(f"epoch {epoch:4d} | loss {train_loss:.4f}")

        return {
            "loss": np.array(loss_hist),
            "val_loss": np.array(val_loss_hist) if val_loss_hist else None,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward_pass(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        if self.output_activation_name == "softmax":
            return np.argmax(proba, axis=1)
        return proba  # for reg/sigmoid outputs


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        if y.ndim > 1 and y.shape[1] > 1:
            y_true = np.argmax(y, axis=1)
        else:
            y_true = y.reshape(-1)
        return float(np.mean(y_pred == y_true))
    


iris = load_iris()
X = iris.data.astype(float)
y = iris.target.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test  = np.hstack([np.ones((X_test.shape[0], 1)),  X_test])

net = ANN(
    input_size=X_train.shape[1],
    output_size=3,
    net_length=2,
    net_width=[10, 10, 10],
    hidden_activation="relu",
    output_activation="softmax",
    loss="cross_entropy",
    seed=0,
)

hist = net.train(
    X_train, y_train,
    lr=0.1, epochs=300, batch_size=32,
    verbose=1
)

print("Train accuracy:", net.score(X_train, y_train))
print("Test  accuracy:", net.score(X_test,  y_test))
