import sys
import numpy

### Check if we want to load weights or save new ones.
load=False
if (len(sys.argv) >= 2 and sys.argv[1] == 'l'):
    load=True

class NeuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who (ih = input->hidden , ho = hidden->output)
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc  
        self.wih = numpy.random.uniform(-0.5, +0.5, size=(self.hnodes, self.inodes))
        self.who = numpy.random.uniform(-0.5, +0.5, size=(self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: 1/(1 + numpy.exp(-x))

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def load(self):
        print("Loading weights ...")
        self.wih = numpy.loadtxt('wih.csv', delimiter=',') 
        self.who = numpy.loadtxt('who.csv', delimiter=',')

    def save(self):
        numpy.savetxt('wih.csv', self.wih, delimiter=',') 
        numpy.savetxt('who.csv', self.who, delimiter=',')
        print("Weights were saved.")

    def log(self):
        print(self.wih)
        print(self.who)

# number of input, hidden and output nodes
input_nodes = 2
hidden_nodes = 16
output_nodes = 2

# learning rate
learning_rate = 0.1

# create instance of neural network
n = NeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# training data
# 0 XOR 0 = 0
# 0 XOR 1 = 1
# 1 XOR 0 = 1
# 1 XOR 1 = 0
training_data_input = [[0, 0], [0, 1], [1, 0], [1, 1]]
training_data_target = [[1, 0], [0, 1], [0, 1], [1, 0]]

# train the neural network, or load previous weights
if (load):
    n.load()
else:
    # epochs is the number of times the training data set is used for training
    epochs = 10000
    for e in range(epochs):
        # go through all items in the training data set
       for i in range(len(training_data_input)):
           n.train(training_data_input[i], training_data_target[i])
       if e % 500 == 0:
           print('Epoch: ', e)

# query the network after it's trained
# then print the output, the index of the highest value corresponds to the label
outputs = n.query([0, 0])
print(outputs)
label = numpy.argmax(outputs)
print("network says ", label)

outputs = n.query([0, 1])
print(outputs)
label = numpy.argmax(outputs)
print("network says ", label)

outputs = n.query([1, 0])
print(outputs)
label = numpy.argmax(outputs)
print("network says ", label)

outputs = n.query([1, 1])
print(outputs)
label = numpy.argmax(outputs)
print("network says ", label)

# save the weights into csv files (if not loaded before).
if (not load):
    n.save()
