import numpy as np
import copy
import matplotlib.pyplot as plt
# GOAL THIS TIME IS TO TRY USING CROSS ENTROPY LOSS INSTEAD OF MSE.
def get_mnist():
    # The code to download the mnist data original came from
    # https://cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html
    
    import gzip
    import numpy as np
    import os
    import struct

    from urllib.request import urlretrieve 

    def load_data(src, num_samples):
        print("Downloading " + src)
        gzfname, h = urlretrieve(src, "./delete.me")
        print("Done.")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x3080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))[0]
                if n != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} entries.".format(num_samples)
                    )
                crow = struct.unpack(">I", gz.read(4))[0]
                ccol = struct.unpack(">I", gz.read(4))[0]
                if crow != 28 or ccol != 28:
                    raise Exception(
                        "Invalid file: expected 28 rows/cols per image."
                    )
                # Read data.
                res = np.frombuffer(
                    gz.read(num_samples * crow * ccol), dtype=np.uint8
                )
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples, crow, ccol)) / 256


    def load_labels(src, num_samples):
        print("Downloading " + src)
        gzfname, h = urlretrieve(src, "./delete.me")
        print("Done.")
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack("I", gz.read(4))
                # Read magic number.
                if n[0] != 0x1080000:
                    raise Exception("Invalid file: unexpected magic number.")
                # Read number of entries.
                n = struct.unpack(">I", gz.read(4))
                if n[0] != num_samples:
                    raise Exception(
                        "Invalid file: expected {0} rows.".format(num_samples)
                    )
                # Read labels.
                res = np.frombuffer(gz.read(num_samples), dtype=np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((num_samples))


    def try_download(data_source, label_source, num_samples):
        data = load_data(data_source, num_samples)
        labels = load_labels(label_source, num_samples)
        return data, labels
    
    # Not sure why, but yann lecun's website does no longer support 
    # simple downloader. (e.g. urlretrieve and wget fail, while curl work)
    # Since not everyone has linux, use a mirror from uni server.
    #     server = 'http://yann.lecun.com/exdb/mnist'
    server = 'https://raw.githubusercontent.com/fgnt/mnist/master'
    
    # URLs for the train image and label data
    url_train_image = f'{server}/train-images-idx3-ubyte.gz'
    url_train_labels = f'{server}/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    print("Downloading train data")
    train_features, train_labels = try_download(url_train_image, url_train_labels, num_train_samples)

    # URLs for the test image and label data
    url_test_image = f'{server}/t10k-images-idx3-ubyte.gz'
    url_test_labels = f'{server}/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    print("Downloading test data")
    test_features, test_labels = try_download(url_test_image, url_test_labels, num_test_samples)
    
    return train_features, train_labels, test_features, test_labels

### step 1 : create perceptron layer class
class PerceptronLayer:
    def __init__(self, arg1, arg2, transfer_func="relu"):
        if np.isscalar(arg1) and np.isscalar(arg2):
            # arg 1 is num inputs, arg 2 is num outputs
            self.inputs = arg1
            self.outputs = arg2

            self.weights = np.random.uniform(-1, 1, (self.outputs, self.inputs))

            self.bias = np.random.uniform(-1, 1, (self.outputs, 1))

        else:
            # arg1 is weight matrix, arg2 is bias vector
            self.weights = np.array(arg1)
            self.bias = np.array(arg2)

            if self.bias.ndim == 1:
                self.bias = self.bias.reshape(-1, 1)

            self.outputs, self.inputs = self.weights.shape

        self.transfer_func_name = transfer_func

        self.last_input = None
        self.z = None

    # --- Private Transfer Functions ---
    def __do_transfer(self, n):
        if self.transfer_func_name == "relu":
            return self.__relu(n)
        elif self.transfer_func_name == "softmax":
            return self.__softmax(n)
        pass

    def __relu(self, n):
        return np.maximum(0, n)

    def forward(self, p):
        p = np.array(p)

        if p.ndim == 1:
            p = p.reshape(-1, 1)

        if p.shape[0] != self.inputs:
            return "Error: vector size is incorrect"

        bias_vec = np.array(self.bias).reshape(-1, 1)
        n = np.dot(self.weights, p) + bias_vec

        self.last_input = p
        self.z = n

        self.a = self.__do_transfer(n)

        return self.a
    
    def __softmax(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
        shifted = x - np.max(x, axis=0, keepdims=True)
        shifted = np.clip(shifted, -500.0, 500.0)
        e_x = np.exp(shifted)
        denom = np.sum(e_x, axis=0, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return e_x / denom
    def __activation_derivative(self, x):
        if self.transfer_func_name == "relu":
            return np.where(x > 0, 1, 0)
        elif self.transfer_func_name == "softmax":
            return 1

    def backward(self, d_out, learning_rate=0.01):
        self.weights = np.asanyarray(self.weights)
        self.bias = np.asanyarray(self.bias)

        delta = d_out * self.__activation_derivative(self.z)
        dW = np.dot(delta, self.last_input.T)
        db = np.sum(delta, axis=1, keepdims=True)
        d_prev = np.dot(self.weights.T, delta)
        self.weights -= learning_rate * dW
        self.bias = self.bias.reshape(db.shape) - (learning_rate * db)
        return d_prev

    def printW(self):
        print("Weights: ")
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                print(self.weights[i][j], end=" ")
            print()
        print("Bias: ", self.bias)


class NeuralNetwork:
    def __init__(self, neuronCount, transfers):
        #neuronCount[-1] is num outputs, neuronCount[0] is num inputs
        num_weight_layers = len(neuronCount) - 1

        if len(transfers) != num_weight_layers:
            raise ValueError(
                f"Expected {num_weight_layers} transfer functions, got {len(transfers)}"
            )

        self.layers = []
        for i in range(num_weight_layers):
            std_dev = np.sqrt(2.0 / neuronCount[i])
            w = np.random.normal(0, std_dev, (neuronCount[i + 1], neuronCount[i]))
            b = np.zeros((neuronCount[i + 1], 1))
            self.layers.append(PerceptronLayer(w, b, transfers[i]))
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, d_out, learning_rate=0.01):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out, learning_rate)
    
train_features, train_labels, test_features, test_labels = get_mnist()

print("train_features:", train_features.shape, train_features.dtype, train_features.min(), train_features.max())
print("train_labels:", train_labels.shape, train_labels.dtype, np.unique(train_labels)[:10])
print("test_features:", test_features.shape, test_features.dtype, test_features.min(), test_features.max())
print("test_labels:", test_labels.shape, test_labels.dtype, np.unique(test_labels)[:10])

print("one sample image shape:", train_features[0].shape)
print("one sample label:", train_labels[0])
my_train_features = train_features.reshape(train_features.shape[0], -1)
print("reshaped train features:", my_train_features.shape)
my_test_features = test_features.reshape(test_features.shape[0], -1)
print("reshaped test features:", my_test_features.shape)

def to_one_hot(labels, num_classes=10):
    # labels shape: (N,) with ints in [0, num_classes-1]
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot
my_train_labels = to_one_hot(train_labels)
my_test_labels = to_one_hot(test_labels)
print("one hot train labels:", my_train_labels.shape, my_train_labels.dtype, np.unique(my_train_labels, axis=0))
print("one hot test labels:", my_test_labels.shape, my_test_labels.dtype, np.unique(my_test_labels, axis=0))


def predict_class(net, x_row):
    return int(np.argmax(net.forward(x_row.reshape(-1, 1))))


def accuracy(net, features, labels):
    correct = 0
    for x_row, y in zip(features, labels):
        correct += int(predict_class(net, x_row) == int(y))
    return correct / len(labels)

def train_once(num_layers, num_neurons):
    max_epochs = 200
    min_delta = 1e-4
    learning_rate = 0.01
    patience = 3

    train_limit = 10000
    test_limit = 2000

    hidden_sizes = [num_neurons] * num_layers
    neuron_count = [784] + hidden_sizes + [10]
    transfers = ["relu"] * num_layers + ["softmax"]
    net = NeuralNetwork(neuron_count, transfers)

    X_train = my_train_features[:train_limit]
    y_train_oh = my_train_labels[:train_limit]
    X_test = my_test_features[:test_limit]
    y_test = test_labels[:test_limit]

    ce_history = []
    stagnant = 0

    for epoch in range(max_epochs):
        epoch_ce_loss = 0.0

        for x_row, y_row in zip(X_train, y_train_oh):
            x_col = x_row.reshape(-1, 1)
            y_col = y_row.reshape(-1, 1)

            out = net.forward(x_col)
            error = out - y_col  # gradient for KL divergence with one hot.
            epoch_ce_loss += -np.sum(y_col * np.log(np.clip(out, 1e-15, 1.0)))
            net.backward(error, learning_rate)

        ce_loss = epoch_ce_loss / len(X_train)
        ce_history.append(ce_loss)

        if epoch > 0:
            if abs(ce_history[-1] - ce_history[-2]) < min_delta:
                stagnant += 1
            else:
                stagnant = 0
            if stagnant >= patience:
                break

    final_test_accuracy = accuracy(net, X_test, y_test)

    return {
        "num_layers": num_layers,
        "num_neurons": num_neurons,
        "epochs_to_converge": len(ce_history),
        "final_ce_loss": ce_history[-1],
        "final_test_accuracy": final_test_accuracy,
        "ce_history": ce_history,
    }
results = []
for num_layers in range(1, 4):
    for num_neurons in [10, 30, 50, 70]:
        result = train_once(num_layers, num_neurons)
        print(
            f"layers={result['num_layers']} neurons={result['num_neurons']} "
            f"epochs={result['epochs_to_converge']} "
            f"test_acc={result['final_test_accuracy']:.4f} "
            f"final_ce={result['final_ce_loss']:.6f}"
        )
        results.append(result)
# making the heatmaps
layer_values = sorted({r["num_layers"] for r in results})
neuron_values = sorted({r["num_neurons"] for r in results})

acc_map = np.zeros((len(layer_values), len(neuron_values)))
epoch_map = np.zeros((len(layer_values), len(neuron_values)))

for r in results:
    i = layer_values.index(r["num_layers"])
    j = neuron_values.index(r["num_neurons"])
    acc_map[i, j] = r["final_test_accuracy"]
    epoch_map[i, j] = r["epochs_to_converge"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

im1 = ax1.imshow(acc_map, cmap="viridis", aspect="auto")
ax1.set_title("Final Test Accuracy")
ax1.set_xlabel("Neurons per Hidden Layer")
ax1.set_ylabel("Hidden Layer Count")
ax1.set_xticks(range(len(neuron_values)))
ax1.set_xticklabels(neuron_values)
ax1.set_yticks(range(len(layer_values)))
ax1.set_yticklabels(layer_values)
for i in range(len(layer_values)):
    for j in range(len(neuron_values)):
        ax1.text(j, i, f"{acc_map[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(epoch_map, cmap="magma", aspect="auto")
ax2.set_title("Epochs to Converge")
ax2.set_xlabel("Neurons per Hidden Layer")
ax2.set_ylabel("Hidden Layer Count")
ax2.set_xticks(range(len(neuron_values)))
ax2.set_xticklabels(neuron_values)
ax2.set_yticks(range(len(layer_values)))
ax2.set_yticklabels(layer_values)
for i in range(len(layer_values)):
    for j in range(len(neuron_values)):
        ax2.text(j, i, f"{int(epoch_map[i, j])}", ha="center", va="center", color="white", fontsize=8)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("src/task2/heatmaps_layers_vs_neurons.png", dpi=150)
plt.close()
