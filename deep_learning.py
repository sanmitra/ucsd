from __future__          import division
from scipy.stats         import zscore
import matplotlib.pyplot as plt
import numpy             as np
import array
import math
import struct
import random

def read_mnist(images_file, labels_file): 
    f1 = open(labels_file, 'rb')
    magic_number, size = struct.unpack(">II", f1.read(8))
    labels = array.array("b", f1.read())
    f1.close()
    
    f2 = open(images_file, 'rb')
    magic_number, size, rows, cols = struct.unpack(">IIII", f2.read(16))
    raw_images = array.array("B", f2.read())
    f2.close()

    N = len(labels)
    images = np.zeros((N, rows*cols), dtype=np.uint8)
    for i in range(N):
        images[i] = np.array(raw_images[ i*rows*cols : (i+1)*rows*cols ])

    return images, labels

# Read Training data.
TRAIN_IMAGES  = "C:\\Users\\oop\\Desktop\\Winter 2016\\train-images.idx3-ubyte"
TRAIN_LABELS  = "C:\\Users\\oop\\Desktop\\Winter 2016\\train-labels.idx1-ubyte"
images_train, labels_train = read_mnist(TRAIN_IMAGES, TRAIN_LABELS)
#images_train, labels_train = images_train[:20000], labels_train[:20000]

# Read Test data.
TEST_IMAGES = "C:\\Users\\oop\\Desktop\\Winter 2016\\t10k-images.idx3-ubyte"
TEST_LABELS  = "C:\\Users\\oop\\Desktop\\Winter 2016\\t10k-labels.idx1-ubyte"
images_test, labels_test = read_mnist(TEST_IMAGES, TEST_LABELS)
#images_test, labels_test = images_test[:2000], labels_test[:2000]


# In[11]:

def sigmoid(x):
    if type(x) is np.ndarray or type(x) is list:
        return np.array([sigmoid(ele) for ele in x])
    else:
        return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivate(x):
    if type(x) is np.ndarray or type(x) is list:
        return np.array([sigmoid_derivate(ele) for ele in x])
    else:
        a = sigmoid(x)
        return a * (1-a)


# In[92]:

X = zscore(images_train, axis=1)
Y = [np.array([1 if i == label else 0 for i in range(10)]) for label in labels_train]
X_test = zscore(images_test, axis=1)
Y_test = [np.array([1 if i == label else 0 for i in range(10)]) for label in labels_test]


# In[97]:

class MultiLayerNeuralNetwork:
    
    def __init__(self, inputs, outputs, learning_rate, layers, 
                 activation_fn, activation_derivative_fn, 
                 validation_size, *args, **kwargs):
        """
        TODO: Add doc.
        """
        train_data_size = len(inputs) - validation_size
        self.inputs = inputs[:train_data_size]
        self.outputs = outputs[:train_data_size]
        self.cross_validation_inputs = inputs[train_data_size:]
        self.cross_validation_outputs = outputs[train_data_size:]
        self.learning_rate = learning_rate
        self.layers = layers
        self.activation_fn = activation_fn
        self.activation_derivative_fn = activation_derivative_fn
    
    def get_random_weights(self):
        weights = []
        for i in range(len(self.layers)-1):
            weights.append(np.random.random((self.layers[i]+1, 
                                             self.layers[i+1])))
        return weights
    
    def get_gradients(self, X, Y, weights):
        # Forward propogation.
        a,z = self.get_network_output(X, weights)
        
        # Backward error propogation.
        deltas = []
        deltas.append((a[-1] - Y))
        for l in reversed(range(1, len(self.layers)-1)):
            deltas.append(np.dot(weights[l], deltas[-1])*self.activation_derivative_fn(z[l]))
        deltas.reverse()
        
        gradients = []
        for i in range(len(weights)):
            if i != (len(weights)-1):
                deltas[i] = deltas[i][1:]
            gradients.append(np.dot(np.atleast_2d(a[i]).transpose(), 
                                    np.atleast_2d(deltas[i])))
        return gradients
            
    def train(self, weights=None, iterations=400000):
        """
        Trains the data using multilayered.
        """
        weights     = weights or self.get_random_weights()
        plot_points = [1,100,1000,10000,100000,400000]
        train_error = []
        test_error  = []
        k = 0
        # On-line Learning.
        for itr in range(1, iterations+1):
            i = random.randint(0,len(self.inputs)-1)
            X = np.insert(self.inputs[i], 0, 1)
            Y = self.outputs[i]
            gradients = self.get_gradients(X,Y,weights)
            # Use gradient descent algorithm to update
            # accordingly due to error derivatives.
            for i in range(len(weights)):
                weights[i] = weights[i] - self.learning_rate * gradients[i]
            
            """
            if itr == plot_points[k]:
                train_error.append(self.test(self.inputs,
                                             self.outputs,
                                             weights)
                                  )
                test_error.append(self.test(self.cross_validation_inputs, 
                                            self.cross_validation_outputs, 
                                            weights)
                                 )
                k += 1
            """
        self.weights     = weights
        self.train_error = train_error
        self.test_error  = test_error
        
        return weights
        
    def get_network_output(self, X, weights):
        """
        Calculates the output at each layer of the network.
        """
        a = [X]
        z = [X]
        for l in range(len(self.layers)-1):
            zl = np.dot(weights[l].transpose(), z[l])
            if l == (len(self.layers)-2):
                output = np.vectorize(math.exp)(zl)
                output = output / output.sum()
                a.append(output)
                z.append(zl)
            else:
                z.append(np.insert(zl,0,1))
                a.append(np.insert(self.activation_fn(zl),0,1))
        return a,z
    
    def cross_entropy(self, weights):
        entropy = 0
        for i in range(len(self.inputs)):
            a,z = self.get_network_output(np.insert(self.inputs[i], 0, 1), 
                                          weights)
            y = a[-1]
            t = self.outputs[i]
            entropy += np.dot(t, np.vectorize(math.log)(y))
        return -entropy
    
    def test(self, test_input, test_output, weights):
        error = 0
        for i in range(len(test_input)):
            X = np.insert(test_input[i], 0, 1)
            T = test_output[i]
            a,z = self.get_network_output(X, weights)
            predicted_digit = a[-1].argmax()
            if T[predicted_digit] != 1:
                error += 1
        return error*100/len(test_input)


# In[98]:

network = MultiLayerNeuralNetwork(inputs=X[:30000],
                                  outputs=Y[:30000],
                                  learning_rate=0.001,
                                  layers=[784,100,10],
                                  activation_fn=sigmoid,
                                  activation_derivative_fn=sigmoid_derivate,
                                  validation_size=10000)
weights = network.get_random_weights()
train_error = []
test_error = []

for i in range(1, 31):
    if i != 0:
        weights = network.train(weights=weights,iterations=10000)
    train_error.append(network.test(network.inputs,
                                    network.outputs,
                                    weights)
                      )
    test_error.append(network.test(network.cross_validation_inputs, 
                                   network.cross_validation_outputs, 
                                   weights)
                     )
    print test_error[-1], train_error[-1]

#final_error = network.test(X_test, Y_test, weights)

