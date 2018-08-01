from edge import Edge
from node import Node
import numpy as np

class NeuralNetwork:
    def __init__(self, inputNodes=np.array([]), hiddenNodes=np.array([]), outputNode=Node(), numInputNodes=5, numHiddenLayers=1, numHiddenLayerNodes=16):
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNode = outputNode
        self.numInputNodes = numInputNodes
        self.numHiddenLayers = numHiddenLayers
        self.numHiddenLayerNodes = numHiddenLayerNodes
        
    def generateInitialNetwork(self, inputNodeValues):
        #Generate input nodes
        for value in inputNodeValues:
            #Create new input nodes
            newNode = Node(inputNodeValue=value)
            self.inputNodes = np.append(self.inputNodes, newNode)
        #Generate hidden nodes
        for x in range(0, self.numHiddenLayerNodes):
            newNode = Node()
            self.hiddenNodes = np.append(self.hiddenNodes, newNode)
        #Generate output node
        self.outputNode = Node()
        
        #Iterate through each input node, and attach edge to each hidden node
        for inputNode in self.inputNodes:
            for hiddenNode in self.hiddenNodes:
                #Create outgoing edge with random weight
                newEdge = Edge(weight=1, prevNode=inputNode, destNode=hiddenNode)
                inputNode.appendOutgoingEdge(newEdge)
                hiddenNode.appendIncomingEdge(newEdge)
                
        #Connect each hidden layer node to the output node
        for hiddenNode in self.hiddenNodes:
            edgeToOutput = Edge(weight=1, prevNode=hiddenNode, destNode=self.outputNode)
            hiddenNode.appendOutgoingEdge(edgeToOutput)
            self.outputNode.appendIncomingEdge(edgeToOutput)
            
    def calculateNodeActivations(self):
        #Start with input nodes
        for inputNode in self.inputNodes:
            inputNode.calculateActivation()
        #Get activation of hidden layer nodes
        for hiddenNode in self.hiddenNodes:
            hiddenNode.calculateActivation()
        #Lastly, calculate output node activation
        self.outputNode.calculateActivation()