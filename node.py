
import numpy as np

class Node:
    def __init__(self, incomingEdges=np.array([]), outgoingEdges=np.array([]), inputNodeValue=None, activation=None, bias=0):
        self.incomingEdges = incomingEdges
        self.outgoingEdges = outgoingEdges
        self.inputNodeValue = inputNodeValue
        self.activation = activation
        self.bias = bias
    
    def calculateActivation(self):
        #Iterate through all incoming edges and sum weights
        self.activation = 0
        for edge in self.incomingEdges:
            self.activation += edge.getWeightActivationProduct()
        #Add bias
        self.activation += self.bias
        #Apply activation function
        #self.activation = np.tanh(self.activation)
        
    #Getters
    def getActivation(self):
        return self.activation
    
    def getInputNodeValue(self):
        return self.inputNodeValue
    
    #Utility functions
    def appendOutgoingEdge(self, edge):
        self.outgoingEdges = np.append(self.outgoingEdges, edge)

    def appendIncomingEdge(self, edge):
        self.incomingEdges = np.append(self.incomingEdges, edge)