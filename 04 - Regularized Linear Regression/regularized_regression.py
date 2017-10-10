import numpy as np

class RegularizedRegression(object):
    def __init__(self):
        pass

    def initialize_weights(self, dim, std_dev=1e-2):
        """
        Initialize the weights of the model. The weights are initialized
        to small random values. Weights are stored in the variable dictionary
        named self.params.

        W: weight vector; has shape (D, 1)
        
        Inputs:
        - dim: (int) The dimension D of the input data.
        - std_dev: (float) Controls the standard deviation of the random values.
        """
        
        self.params = {}
        #############################################################################
        # TODO: Initialize the weight vector to random values with                  #
        # standard deviation determined by the parameter std_dev.                   #
        # Hint: Look up the function numpy.random.randn                             #
        #############################################################################
        self.params['W'] = np.random.randn(dim, 1)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################        
        
        
    def train(self, X, y, poly_order=1, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train linear regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N, 1) containing the ground truth values.
        - poly_order: (integer) determines the polynomial order of your hypothesis function.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        return a list containing the value of the loss function at each training iteration.
        """
        X = self.poly_feature_transform(X, poly_order)
        num_train, dim = X.shape

        # Implement the initialize_weights function.
        self.initialize_weights(dim)

        loss_history = []
        for it in range(num_iters):

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            # Hint: Look up the function numpy.random.choice                        #
            #########################################################################
            indices = np.random.choice(X.shape[0], batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(np.squeeze(loss))

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the model (stored in the dictionary self.params)        #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            self.params['W'] = self.params['W'] - learning_rate*grads['W']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for an iteration of linear regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the ground truth value for X[i].
        - reg: (float) Regularization strength.

        Returns:
        Return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W = self.params['W']
        N, D = X.shape

        #############################################################################
        # TODO: Compute for the prediction value given the current weight vector.   #
        # Store the result in the prediction variable                               #
        #############################################################################
        prediction = np.dot(X, W)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        #############################################################################
        # TODO: Compute for the loss. Include the regularization term.              #
        #############################################################################
        loss = 0.5* np.mean(np.square((prediction-y))) + (reg * 0.5) * np.sum(np.square(W))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        grads = {}
        #############################################################################
        # TODO: Compute the derivatives of the weights. Store the                   #
        # results in the grads dictionary. For example, grads['W'] should store     #
        # the gradient on W, and be a matrix of same size.                          #
        #############################################################################
        
        grads['W'] = np.dot((prediction-y).T,X).T + reg * W
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads
    
    def predict(self, X, poly_order = 1):
        """
        Predict values for test data using linear regression.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - poly_order: Determines the order of the polynomial of the hypothesis function.

        Returns:
        - y: A numpy array of shape (num_test, 1) containing predicted values for the
          test data, where y[i] is the predicted value for the test point X[i].  
        """
        W = self.params['W']
        X = self.poly_feature_transform(X, poly_order)
        #############################################################################
        # TODO: Compute for the predictions of the model on new data using the      #
        # learned weight vectors.                                                   #
        #############################################################################
    
        prediction = np.dot(X, W)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
    
        return prediction

    def poly_feature_transform(self,X,poly_order=1):
        """
        Transforms the input data to match the specified polynomial order.

        Inputs:
        - X: A numpy array of shape (N, D) consisting
             of N samples each of dimension D.
        - poly_order: Determines the order of the polynomial of the hypothesis function.

        Returns:
        - f_transform: A numpy array of shape (N, D + order + 1) representing the transformed
            features following the specified poly_order.
        """
        

        #############################################################################
        # TODO: Append a vector of ones across the dimension of your input data.    #
        # This accounts for the bias or the constant in your hypothesis function.   #
        #############################################################################
        f_transform = np.ones((X.shape[0], X.shape[1]+poly_order)) 
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        #############################################################################
        # TODO: Transform your inputs to the corresponding polynomial with order    #
        # given by the parameter poly_order.                                        #
        #############################################################################  
        for i in range(poly_order + 1):
            f_transform[:,i:(i+1)] = X ** i
			
        #print(f_transform)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return f_transform
    
