import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
   input - fully connected layer - ReLU - fully connected layer - softmax
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    W1: First layer weights; has shape (H, D)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (C, H)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(hidden_size, input_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(output_size, hidden_size)
    self.params['b2'] = np.zeros(output_size)


  def loss(self, X, y=None, reg=0.0):


    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None

    H,_ = W1.shape
    C,_ = W2.shape
    h = np.maximum(0,W1.dot(X.T) +  b1.reshape([H,1]))
    scores = W2.dot(h) + b2.reshape([C,1])
    scores = scores.T

    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None

    # scores is num_examples by num_classes
   
    softmax_loss = np.sum(-np.log( np.exp(scores[np.arange(N),y]) /   np.sum(np.exp(scores),axis = 1)))

    reg_loss = 1/2 * reg * (np.linalg.norm(W1,'fro')**2 + np.linalg.norm(W2,'fro')**2)
    loss = softmax_loss/N + reg_loss

    grads = {}


    #dL/dW2 = day/dW2 * dL/day
    a_exp = np.exp(scores)
    #dLday = pj - 1(y=j)
    prob = a_exp / np.sum(a_exp, axis = 1, keepdims = True) 
    prob[np.arange(N),y] -= 1
    dLday = prob/N # N*C 
    #dLdW2 = relu(W1x+b) =h
    dLdW2 = h # H*N
    grads['W2'] = dLday.T.dot(h.T) + reg*W2 # C*H
    grads['b2'] = np.sum(dLday,axis=0,keepdims=True)
    #dLdW1 = da/dW1 * dh/da * day/dh * dL/day
    daydh = W2.T #H*C
    dLdh = daydh.dot(dLday.T)# H*N

    dLda = (W1.dot(X.T) > 0) * dLdh  # H * N

    dLdW1 = dLda.dot(X) # H*D

    grads['W1'] =  dLdW1 + reg*W1

    grads['b1'] = np.sum(dLda,axis=1,keepdims=True).T

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):

    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in np.arange(num_iters):
      X_batch = None
      y_batch = None

      index = np.random.choice(np.arange(num_train), batch_size)
      X_batch = X[index]
      y_batch = y[index]

       # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      self.params['W1'] = self.params['W1'] - grads['W1']* learning_rate
      self.params['b1'] = self.params['b1'] - grads['b1']* learning_rate
      self.params['W2'] = self.params['W2'] - grads['W2']* learning_rate
      self.params['b2'] = self.params['b2'] - grads['b2']* learning_rate

      if verbose and it % 100 == 0:
        print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):

    y_pred = None

    a = self.params['W1'].dot(X.T) + self.params['b1'].T
    h = np.maximum(0,a)
    ay = self.params['W2'].dot(h) + self.params['b2'].T
    y_pred = np.argmax(ay,axis = 0)
    # print('y_pred',y_pred.shape)


    return y_pred


