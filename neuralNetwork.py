import numpy as numpy
import scipy.special
import matplotlib.pyplot
#% matplotlib inline


class neuralNetwork:

    def __init__(self, inodes, hnodes, onodes, learningRate):
        self.indodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        self.lr = learningRate

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.indodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, input_list, targets_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs

        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs), numpy.transpose(hidden_outputs)))

        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

data_file = open("mnist_train_100.csv")
data_list = data_file.readlines()

for record in data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99

    n.train(inputs, targets)

# onodes = 10
# image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
# matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

# scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
# print(scaled_input)



