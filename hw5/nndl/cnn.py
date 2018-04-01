import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    #conv - relu - 2x2 max pool - affine - relu - affine - softmax

    #input dimension
    C,H,W = input_dim

    HH = (H - 2)//2 + 1 # 2*2 pooling with stride 2 
    WW = (W - 2)//2 + 1



    #conv - relu - pool: conv_relu_pool_forward(x, w, b, conv_param, pool_param)
    # w: (F, C, HH, WW)
    self.params['W1'] = np.random.normal(0,weight_scale,size = (num_filters,C, filter_size,filter_size))
    self.params['b1'] = np.zeros(num_filters)

    # affine - relu: affine_relu_forward(x, w, b):

    self.params['W2'] = np.random.normal(0,weight_scale,size = (num_filters*HH*WW, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)

    # affine - softmax
    self.params['W3'] = np.random.normal(0,weight_scale,size = (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #

    conv1,conv_cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    h1,fc_cache1 = affine_relu_forward(conv1, W2, b2)
    scores,fc_cache2 = affine_forward(h1, W3, b3)


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    loss, dscores = softmax_loss(scores, y)

    dh1, grads['W3'], grads['b3'] = affine_backward(dscores, fc_cache2)

    dconv1, grads['W2'], grads['b2'] = affine_relu_backward(dh1, fc_cache1)

    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dconv1, conv_cache1)

    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    grads['W1'] += grads['W1'] * self.reg
    grads['W2'] += grads['W2'] * self.reg
    grads['W3'] += grads['W3'] * self.reg
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
  
  
class TestConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Initialize the weights and biases of a three layer CNN. To initialize:
    #     - the biases should be initialized to zeros.
    #     - the weights should be initialized to a matrix with entries
    #         drawn from a Gaussian distribution with zero mean and 
    #         standard deviation given by weight_scale.
    # ================================================================ #

    #(conv - relu - 2x2 max pool)*3 - affine*3 - softmax

    #input dimension
    C,H,W = input_dim

    HH = (H - 2)//2 + 1 # 2*2 pooling with stride 2 
    WW = (W - 2)//2 + 1



    #conv - relu - pool: conv_relu_pool_forward(x, w, b, conv_param, pool_param)
    # w: (F, C, HH, WW)
    self.params['W1'] = np.random.normal(0,weight_scale,size = (num_filters,C, filter_size,filter_size))
    self.params['b1'] = np.zeros(num_filters)

    #conv - relu - pool: conv_relu_pool_forward(x, w, b, conv_param, pool_param)
    # w: (F, C, HH, WW)
    self.params['W2'] = np.random.normal(0,weight_scale,size = (num_filters,num_filters, filter_size,filter_size))
    self.params['b2'] = np.zeros(num_filters)

    #conv - relu - pool: conv_relu_pool_forward(x, w, b, conv_param, pool_param)
    # w: (F, C, HH, WW)
    self.params['W3'] = np.random.normal(0,weight_scale,size = (num_filters,C, filter_size,filter_size))
    self.params['b3'] = np.zeros(num_filters)





    # affine - relu: affine_relu_forward(x, w, b):

    self.params['W4'] = np.random.normal(0,weight_scale,size = (num_filters*HH*WW, hidden_dim))
    self.params['b4'] = np.zeros(hidden_dim)

    # affine - (softmax)
    self.params['W5'] = np.random.normal(0,weight_scale,size = (hidden_dim, num_classes))
    self.params['b5'] = np.zeros(num_classes)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of the three layer CNN.  Store the output
    #   scores as the variable "scores".
    # ================================================================ #

    conv1,conv_cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    
    h1,fc_cache1 = affine_relu_forward(conv1, W2, b2)
    scores,fc_cache2 = affine_forward(h1, W3, b3)


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    if y is None:
      return scores
    
    loss, grads = 0, {}
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of the three layer CNN.  Store the grads
    #   in the grads dictionary, exactly as before (i.e., the gradient of 
    #   self.params[k] will be grads[k]).  Store the loss as "loss", and
    #   don't forget to add regularization on ALL weight matrices.
    # ================================================================ #
    loss, dscores = softmax_loss(scores, y)

    dh1, grads['W3'], grads['b3'] = affine_backward(dscores, fc_cache2)

    dconv1, grads['W2'], grads['b2'] = affine_relu_backward(dh1, fc_cache1)

    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(dconv1, conv_cache1)


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grads
