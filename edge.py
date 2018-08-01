from node import Node
import numpy as np

class Edge:
    weight = 1
    prevNode = None
    destNode = None
    
    def __init__(self, weight=1, prevNode=None, destNode=None):
        self.weight = weight
        self.prevNode = prevNode
        self.destNode = destNode
    
    def getWeight(self):
        return self.weight
    
    def getWeightActivationProduct(self):
        #If not input node, calc activation. Otherwise, return input value
        if(self.prevNode.getInputNodeValue() == None):
            
            return self.weight * self.prevNode.getActivation()
        return self.weight * self.prevNode.getInputNodeValue()