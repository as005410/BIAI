#import the necessary packages
import numpy as np


class NeuralNetwork:
	#constructor of NN
	#layers->architecture of the network
	#alpha->learning rate
    def __init__(self, layers, alpha=0.1):
      # initialize the list of weight matrices
      self.W = []
      #number of layers
      self.layers = layers
      #learning rate
      self.alpha = alpha 
      #loop through layers(except last 2)
      for i in np.arange(0, len(layers)-2):
        #randomly fill 2d arrays for weights
        weights = np.random.randn(layers[i] + 1,layers[i + 1] + 1)
        self.W.append(weights / np.sqrt(layers[i]))
      #last layer without bias
      weights = np.random.randn(layers[-2] + 1, layers[-1])
      self.W.append(weights / np.sqrt(layers[-2]))

    # construct and return a string that represents the network acrhitecture
    def __repr__(self):
      return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    #compute sigmoid activation function
    def sigmoid(self, x):
      return 1.0/(1+np.exp(-x))

    # compute the derivative
    def sigmoid_deriv(self, x):
      return x*(1-x)

    #return activations
    def train_partial(self, x, y):
      
      #feedforward
      A = [np.atleast_2d(x)]
      #loop through layers
      for layer in np.arange(0, len(self.W)):
        #dot product of activation and weight
        net = A[layer].dot(self.W[layer])
        #activate with function
        out = self.sigmoid(net)
        #add to activation list
        A.append(out)

      #backpropagation
      #computing error
      error = A[-1] - y
      #adding to delta list
      D = [error * self.sigmoid_deriv(A[-1])]

      #loop over layers backwards
      for layer in np.arange(len(A) - 2, 0, -1) :
        #computing deltas
        delta = D[-1].dot(self.W[layer].T)
        delta = delta * self.sigmoid_deriv(A[layer])
        #adding to the list
        D.append(delta)
      #reverse deltas
      D = D[::-1]
      #loop over layers to update weights
      for layer in np.arange(0, len(self.W)):
        #calculate new weights(dot product of activation and delta and multiply by aplha)
        self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])


    #function for training
    #X = training data
    #Y = result of input data
    #epochs = number of passes
    #displayUpdate = how many epoches to print
    def train(self, X, Y, epochs=1000, displayUpdate=100):
      #add column for bias
      X = np.c_[X, np.ones((X.shape[0]))]
		
      # loop over the epochs and train network
      for epoch in np.arange(0, epochs):
        #loop over every data point
        for (x,target) in zip(X,Y):
          self.train_partial(x, target)
        #print every 100'th epoch
        if epoch == 0 or (epoch+1) % displayUpdate == 0:
          loss =  self.calculate_loss(X,Y)
          print("epoch = {}, loss = {:.7f}".format(epoch+1,loss))
    
    #function to predict 
    def predict(self, X, addBias=True):
      prediction = np.atleast_2d(X)
      #add bias
      if addBias:
        prediction = np.c_[prediction, np.ones((prediction.shape[0]))]
      #loop over layers
      for layer in np.arange(0, len(self.W)):
        #computing prediction
        prediction = self.sigmoid(np.dot(prediction, self.W[layer]))

      return prediction

    #calculate loss
    def calculate_loss(self, X, targets):
      #make list with our targets
      targets = np.atleast_2d(targets)
      predictions = self.predict(X, addBias=False)
      #calculating error
      loss = 0.5 * np.sum((predictions - targets) ** 2)

      return loss
