#!/usr/bin/env python3
import numpy as np
np.random.seed(0)
'''
Neural net from scratch
_______________________________
Aka Drawing a line of best fit:
_______________________________
Aka line equation mx+b
_______________________________
Author: Me
Date: 19_02_23
vid:4:min 10:40
 
'''
import sys
import numpy as np
import matplotlib

'''
###
NOTE
* Shape of array from out(leftmost) to in(rightmost)
* Tensnsor an object that can be represented as an array.
###
# A list basic memthod of implementing input * weights + bias for 4 neurons to 3 neurons
inputs = [1,2,3,2.5] 
weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5,-0.91,0.26,-0.5]
weights3 = [-0.26 ,-0.27,0.17,0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5 

output = [inputs[0]*weights1[0] +inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+bias1,
          inputs[0]*weights2[0] +inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bias2,
          inputs[0]*weights3[0] +inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bias3]
print(output)
#input*weight+bias
'''
# A loop memthod of implementing input * weights + bias 4 neurons to 3 neurons
inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26 ,-0.27,0.17,0.87]]
biases = [2,3,0.5]
print(f"The addition: {inputs+biases}")

output = np.dot(weights, inputs) + biases

print(output)

'''
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26 ,-0.27,0.17,0.87]]
biases = [2,3,0.5]
'''
'''
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights,biases):
	neuron_output = 0
	for n_input, weight in zip(inputs, neuron_weights):
		neuron_output += n_input*weight
	neuron_output +=neuron_bias
	layer_outputs.append(neuron_output)

print(layer_outputs)
'''

class layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.randn(n_input,n_neurons)
        #initialize random weights based on inputs size and No neurons returns matrix shape(n_inputs,n_nurons)
        self.biases = np.zeros(1,n_neurons)
        #init random biases of the shape 1 and n_neurons
        pass
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights)+bias
        pass
x = [[1,2,3,2.5],
     [2.0,5.0,-1.0,2.0]
     [-1.5,2.7,3.3,-0.8]]
layer1 = layer_Dense(4,5)
layer2 = layer_Dense(5,2)
layer1.forward(x)
layer2.forward(layer1.output)