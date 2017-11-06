import numpy as np

class MultinomialLogisticRegression(object):
    def __init__(self, input_dim, num_classes, std_dev=1e-2):
        self.initialize_weights(input_dim, num_classes, std_dev)

    def initialize_weights(self, input_dim, num_classes, std_dev=1e-2):
        """
        Initialize the weights of the model. The weights are initialized
        to small random values. Weights are stored in the variable dictionary
        named self.params.

        W: weight vector; has shape (D, C)
        b: bias vector; has shape (C,)
        
        Inputs:
        - input_dim: (int) The dimension D of the input data.
        - std_dev: (float) Controls the standard deviation of the random values.
        """
        
        self.params = {}
        #############################################################################
        # TODO: Initialize the weight and bias.                                     #
        #############################################################################
        self.params['W'] = std_dev*np.random.randn(input_dim, 1)
        self.params['b'] = 0
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
        
        
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train Linear Regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N, 1) containing the ground truth values.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        loss_history = []
        for it in range(num_iters):

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            indices = np.random.choice(num_train,batch_size)
            X_batch = X[indices,:]
            y_batch = y[indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)

            if it % 100 == 0:
                loss_history.append(np.squeeze(loss))

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the model (stored in the dictionary self.params)        #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            self.params['W'] = self.params['W'] + learning_rate*grads['W']
            self.params['b'] = self.params['b'] + learning_rate*grads['b']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def softmax(self,x):
        
        probs = np.exp(x) / np.sum(np.exp(x))

        return probs.T


    def cross_entropy(self,probs, labels):
        N = probs.shape[0]
        
        ce = -np.mean(labels*np.log(N))
        
        return ce
        


    def softmax_cross_entropy_loss(self,x,labels):
        N = x.shape[0]
        probs = self.softmax(x) # turn scores into probs 
        loss = self.cross_entropy(probs,labels)

        dloss = probs.copy()
        #########################################################################
        # TODO: Calculate for the gradients of the loss                         #
        #########################################################################
        
        dloss = (dloss-1)*N # (probs-1)*x; (dloss-1)*x
        
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        
        # the gradient of the loss may already be calculated here
        return loss, dloss

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for an iteration of linear regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the ground truth value for X[i].
        - reg: Regularization strength.

        Returns:
        Return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W, b = self.params['W'], self.params['b']
        N, D = X.shape

        # Compute the forward pass
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        score = np.dot(X, W) + b
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        #############################################################################
        softmax_ce_loss, dloss = self.softmax_cross_entropy_loss(score,y)
        loss = softmax_ce_loss + (reg*0.5)*np.sum(np.square(W))
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the derivatives of the weights and biases. Store the        #
        # results in the grads dictionary. For example, grads['W'] should store     #
        # the gradient on W, and be a matrix of same size.                          #
        #############################################################################
        dW = np.dot(X.T,(y - self.softmax(score).T))/N + reg*W

        db = np.mean(y - self.softmax(score))
        
        grads['W'] = dW
        grads['b'] = db
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads
    
    def predict(self, X, poly_order = 1):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - prediction: A sorted numpy array of shape (num_test,num_classes) containing the probabilities for X[i] belonging to each of the classes
        """
        W, b = self.params['W'], self.params['b']
        scores = X.dot(W) + b
        probs = self.softmax(score)
        prediction = np.array([probs > 0.5]).astype(int)

        return prediction

