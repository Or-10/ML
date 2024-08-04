import numpy as np

# Input features (Hours Slept, Hours Studied)
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)

# Labels (Marks obtained)
y = np.array(([92], [86], [89]), dtype=float)

# Normalize input features and labels
X = X / np.amax(X, axis=0)
y = y / 100

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_grad(x):
    return x * (1 - x)

# Set training parameters
epoch = 1000  # Number of training iterations
eta = 0.2  # Learning rate

# Set the number of neurons
input_neurons = 2  # Number of features in the dataset
hidden_neurons = 3  # Number of neurons in the hidden layer
output_neurons = 1  # Number of neurons in the output layer

# Initialize weights and biases
wh = np.random.uniform(size=(input_neurons, hidden_neurons))  # 2x3
bh = np.random.uniform(size=(1, hidden_neurons))  # 1x3
wout = np.random.uniform(size=(hidden_neurons, output_neurons))  # 3x1
bout = np.random.uniform(size=(1, output_neurons))  # 1x1

# Training the neural network
for i in range(epoch):
    # Forward propagation
    h_ip = np.dot(X, wh) + bh  # Dot product + Bias
    h_act = sigmoid(h_ip)  # Activation function
    o_ip = np.dot(h_act, wout) + bout  # Dot product + Bias
    output = sigmoid(o_ip)  # Activation function

    # Backpropagation
    # Error at output layer
    Eo = y - output  # Error at output
    outgrad = sigmoid_grad(output)
    d_output = Eo * outgrad  # Error gradient at output layer

    # Error at hidden layer
    Eh = d_output.dot(wout.T)  # Error at hidden layer
    hiddengrad = sigmoid_grad(h_act)
    d_hidden = Eh * hiddengrad  # Error gradient at hidden layer

    # Update weights and biases
    wout += h_act.T.dot(d_output) * eta  # Update output layer weights
    wh += X.T.dot(d_hidden) * eta  # Update hidden layer weights

print("Normalized Input: \n" , str(X))
print("\nActual Output: \n" , str(y))
print("\nPredicted Output: \n", output)
