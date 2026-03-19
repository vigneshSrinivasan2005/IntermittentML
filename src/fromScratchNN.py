import numpy as np
from abc import ABC, abstractmethod

# --- Core Neural Network Components ---

class PerceptronLayer:
    def __init__(self, arg1, arg2, transfer_func="relu"):
        if np.isscalar(arg1) and np.isscalar(arg2):
            self.inputs = arg1
            self.outputs = arg2

            # He Initialization for ReLU, Xavier for Linear/Sigmoid
            if transfer_func == "relu":
                std_dev = np.sqrt(2.0 / self.inputs)
            else:
                std_dev = np.sqrt(1.0 / self.inputs)

            self.weights = np.random.normal(0, std_dev, (self.outputs, self.inputs))
            self.bias = np.zeros((self.outputs, 1))
        else:
            self.weights = np.array(arg1)
            self.bias = np.array(arg2)
            if self.bias.ndim == 1:
                self.bias = self.bias.reshape(-1, 1)
            self.outputs, self.inputs = self.weights.shape

        self.transfer_func_name = transfer_func

        self.last_input = None
        self.z = None
        self.weight_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

    def _do_transfer(self, n):
        if self.transfer_func_name == "relu":
            return np.maximum(0, n)
        elif self.transfer_func_name == "linear":
            return n

    def forward(self, p):
        # p is expected to be [features, batch_size]
        p = np.array(p)
        if p.ndim == 1:
            p = p.reshape(-1, 1)

        bias_vec = np.array(self.bias).reshape(-1, 1)
        self.z = np.dot(self.weights, p) + bias_vec
        self.last_input = p
        self.a = self._do_transfer(self.z)
        return self.a

    def _activation_derivative(self, x):
        if self.transfer_func_name == "relu":
            return np.where(x > 0, 1.0, 0.0)
        elif self.transfer_func_name == "linear":
            return np.ones_like(x)

    def backward(self, d_out):
        # delta = d_loss / dz
        # For batch processing, we ensure correct dimensions
        delta = d_out * self._activation_derivative(self.z)
        
        # Gradient with respect to weights and biases, sum over batch
        self.weight_grad = np.dot(delta, self.last_input.T)
        self.bias_grad = np.sum(delta, axis=1, keepdims=True)
        
        # Gradient for previous layer
        d_prev = np.dot(self.weights.T, delta)
        return d_prev

class NeuralNetwork:
    def __init__(self, neuronCount, transfers):
        # neuronCount[-1] is num outputs, neuronCount[0] is num inputs
        num_weight_layers = len(neuronCount) - 1

        if len(transfers) != num_weight_layers:
            raise ValueError(f"Expected {num_weight_layers} transfer functions, got {len(transfers)}")

        self.layers = []
        for i in range(num_weight_layers):
            self.layers.append(PerceptronLayer(neuronCount[i], neuronCount[i + 1], transfers[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
            
    def get_parameters(self):
        params = []
        for layer in self.layers:
            # Append tuple of (value_array, gradient_array)
            # The optimizer will modify value_array in-place
            params.append((layer.weights, layer.weight_grad))
            params.append((layer.bias, layer.bias_grad))
        return params

# --- Custom Optimizer ---

class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [np.zeros_like(p[0]) for p in self.parameters]
        self.v = [np.zeros_like(p[0]) for p in self.parameters]

    def zero_grad(self):
        # Gradients are overwritten directly in layer.backward via assignment
        pass 

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # In-place update
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# --- Faux Tensor and DataLoader ---

class BaseModel(ABC):
    @abstractmethod
    def forward(self, features): pass
    @abstractmethod
    def compute_loss(self, logits, targets): pass
    @abstractmethod
    def initialize_weights(self): pass
    @abstractmethod
    def predict_proba(self, features): pass
    @abstractmethod
    def predict(self, features, threshold=0.5): pass

    def __call__(self, features):
        return self.forward(features)
        
    def parameters(self):
        return self.network.get_parameters()


class IntermittentSalesMLP(BaseModel):
    def __init__(self, input_dim, hidden_1=64, hidden_2=32):
        neuronCount = [input_dim, hidden_1, hidden_2, 1]
        transfers = ["relu", "relu", "linear"]
        self.network = NeuralNetwork(neuronCount, transfers)

    def forward(self, features):
        x = np.array(features)
        # The input batch size is x.shape[0]
        # Neural network expects [features, batch_size]
        x_t = x.T
        out_t = self.network.forward(x_t)
        # Produce output of shape [batch_size, 1]
        return out_t.T

    def compute_loss(self, logits, targets):
        logits_np = np.array(logits)
        targets_np = np.array(targets)

        # PyTorch BCEWithLogitsLoss logic
        loss_val = np.mean(np.maximum(logits_np, 0) - logits_np * targets_np + np.log(1 + np.exp(-np.abs(logits_np))))
        
        # Derivative w.r.t logits for mean reduction
        sig = 1 / (1 + np.exp(-logits_np))
        loss_grad = (sig - targets_np) / len(targets_np)
        
        return loss_val, loss_grad

    def initialize_weights(self):
        pass # Initialization handled accurately inside PerceptronLayer init

    def predict_proba(self, features):
        logits = self.forward(features)
        logits_squeeze = np.squeeze(logits, axis=1) # [batch]
        return 1 / (1 + np.exp(-logits_squeeze))

    def predict(self, features, threshold=0.5):
        probabilities = self.predict_proba(features)
        return (probabilities >= threshold).astype(np.int32)


class WeightedIntermittentSalesMLP(IntermittentSalesMLP):
    def __init__(self, input_dim, hidden_1=64, hidden_2=32, pos_weight=2.0):
        super().__init__(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2)
        self.pos_weight = float(pos_weight)

    def compute_loss(self, logits, targets):
        logits_np = np.array(logits)
        targets_np = np.array(targets)

        sig = 1 / (1 + np.exp(-logits_np))
        sig = np.clip(sig, 1e-15, 1 - 1e-15)
        loss_val = np.mean(-self.pos_weight * targets_np * np.log(sig) - (1 - targets_np) * np.log(1 - sig))

        # Derivative
        grad_i = sig * (1 - targets_np + self.pos_weight * targets_np) - self.pos_weight * targets_np
        loss_grad = grad_i / len(targets_np)
        
        return loss_val, loss_grad


class WAPEIntermittentSalesMLP(IntermittentSalesMLP):
    def __init__(self, input_dim, hidden_1=64, hidden_2=32):
        super().__init__(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2)

    def compute_loss(self, logits, targets):
        logits_np = np.array(logits)
        targets_np = np.array(targets)

        predictions = 1 / (1 + np.exp(-logits_np))
        numerator = np.sum(np.abs(targets_np - predictions))
        denominator = np.maximum(np.sum(np.abs(targets_np)), 1e-6)
        loss_val = float(numerator / denominator)

        # Derivative (Note: no len(targets) scaling needed as denominator is already across batch sum)
        grad_p_i = np.where(predictions > targets_np, 1.0, -1.0) / denominator
        loss_grad = grad_p_i * predictions * (1 - predictions)

        return loss_val, loss_grad


class DynamicWeightedIntermittentSalesMLP(WeightedIntermittentSalesMLP):
    def __init__(self, input_dim, hidden_1=64, hidden_2=32):
        super().__init__(input_dim=input_dim, hidden_1=hidden_1, hidden_2=hidden_2, pos_weight=1.0)
        
    def set_pos_weight_from_targets(self, y_train):
        y_train_np = np.array(y_train)
        total_ones = np.sum(y_train_np)
        total_zeros = len(y_train_np) - total_ones

        if total_ones <= 0:
            dynamic_weight = 1.0
        else:
            dynamic_weight = total_zeros / total_ones

        print(f"Calculated Dynamic pos_weight: {dynamic_weight:.4f}")
        self.pos_weight = float(dynamic_weight)

