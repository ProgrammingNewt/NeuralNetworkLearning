import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

#function to generate a spiral dataset
def create_spiral_data(points, classes):
    X = np.zeros((points*classes, 2))  #data matrix where each row is a single example
    y = np.zeros(points*classes, dtype='uint8')  
    
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2  
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
        
    return X, y

#dense layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

#reLU activation, makes weight either 0 or max, no negative
class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

#softmx - converts real values into probabilities
class Activation_Softmax:
    
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

#entropy loss class
class Loss_CategoricalCrossentropy:
    
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

#optimizer class using stochastic gradient decent as it is better for bigger data   
#works well for spiral since many points are redundant
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

#to visualize spiral
def visualize_spiral(X, y, title="Spiral Dataset"):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
    plt.title(title)
    plt.show()

#function to predict using the model and visualize these predictions
def visualize_predictions(X, model_layers):
    
    #forward pass through the model
    input_data = X
    for layer in model_layers:
        layer.forward(input_data)
        input_data = layer.output
    predictions = np.argmax(input_data, axis=1)
    visualize_spiral(X, predictions, title="Model Predictions")


X, y = create_spiral_data(100, 3)

#layers
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()

#vizualizaion before training
model_layers = [dense1, activation1, dense2, activation2]
visualize_predictions(X, model_layers)

#loss function and optimizer
loss_function = Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=1.0)

#training loop
for epoch in range(15001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    
    loss = loss_function.forward(activation2.output, y)
    
    #backwards pass
    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    #update weights and biases
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

#vizualize after training complete
visualize_predictions(X, model_layers)
