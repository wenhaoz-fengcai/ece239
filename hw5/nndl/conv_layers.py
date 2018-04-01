import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #

  # get all the parameters
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  H_ = int(1 + (H + 2 * pad - HH) / stride)
  W_ = int(1 + (W + 2 * pad - WW) / stride)

  # pad x   
  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=(0,0))

  #compute out 
  out = np.zeros([N,F,H_,W_])
  for n in np.arange(N):
    for f in np.arange(F):
      for width in np.arange(W_):
        for height in np.arange(H_):
          out[n,f,height,width] = np.sum(x_pad[n, 
                                            :, 
                                            height*stride:height*stride+HH, 
                                            width*stride:width*stride+WW] * w[f, :, :, :]) + b[f]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  N,C,H,W = x.shape
  F,C,HH,WW = w.shape
  # build empty output array
  dxpad = np.zeros_like(xpad)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  # dw = xpad[]* dout 
  # dx = w * xout 
  for n in range(N):
    for f in range(F):
      db[f] += np.sum(dout[n,f,:,:])
      for height in range(out_height):
        for width in range(out_width):
          dw[f] +=  xpad[n, :,  height*stride:height*stride+HH, width*stride:width*stride+WW] * dout[n,f,height,width]
          dxpad[n, :,  height*stride:height*stride+HH, width*stride:width*stride+WW] += w[f]* dout[n,f,height,width]
  dx = dxpad[:,:,pad:-pad,pad:-pad]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  # parameters 
  N, C, H, W = x.shape
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  H_ = int(1 + (H - HH) / stride)
  W_ = int(1 + (W - WW) / stride)

  # empty output
  out = np.zeros([N,C,H_,W_])


  for n in range(N):
    for c in range(C):
      for h in range(H_):
        for w in range(W_):
          out[n,c,h,w] = np.amax(x[n,c,h*stride:h*stride+HH, w*stride:w*stride+WW])

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  
  HH = pool_height
  WW = pool_width
  N, C, H, W = x.shape
  H_ = int(1 + (H - HH) / stride)
  W_ = int(1 + (W - WW) / stride)

  dx = np.zeros_like(x)
  for n in range(N):
    for c in range(C):
      for h in range(H_):
        for w in range(W_):
          x_window = x[n, c, h*stride:h*stride+HH, w*stride:w*stride+WW]
          max_num = np.amax(x_window)
          x_mask = (x_window == max_num)
          dx[n, c, h*stride:h*stride+HH, w*stride:w*stride+WW] += dout[n, c, h, w]*x_mask

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  # get parameters
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  # reshape to 2D to do batchnorm
  N, C, H, W = x.shape
  x = np.reshape(x,(N*H*W, C))

  # copy from 2D batchNorm 
  _, D = x.shape #C = D
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))# C
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))# C

  if mode == 'train':
    mean =np.mean(x,axis=0)# NHW C
    var = np.var(x,axis=0)# NHW C
    out = (gamma*(x-mean)*1./np.sqrt(var+eps)) +beta# NHW C

    running_mean = momentum * running_mean + (1 - momentum) * np.mean(x,axis = 0) # C
    running_var = momentum * running_var + (1 - momentum) * np.var(x,axis = 0)# C

  elif mode == 'test':
    out = gamma*(x-running_mean)/np.sqrt(running_var+eps) +beta# NHW C
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  #update parameters
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  #reshape it back
  mean = mean.reshape(1, C, 1, 1)
  var = var.reshape(1, C, 1, 1)
  x = x.reshape(N, C, H, W)
  out = out.reshape(N,C,H,W)


  cache = (mean, var ,x ,gamma, beta,bn_param)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  mean, var ,x ,gamma, beta,bn_param = cache
  eps = bn_param.get('eps', 1e-5)

  N, C, H, W  = x.shape

  # reshape to 2D
  mean = mean.reshape(C)
  var = var.reshape(C)
  x = x.reshape(N*W*H, C)
  dout = dout.reshape(N*W*H,C)


  # copy 
  NHW, C = dout.shape
  # notation and naming following the note
  e = var + eps
  c = np.sqrt(e)
  b = 1/c
  a = x- mean

  xhat = a*b
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(xhat*dout, axis=0)
  dxhat = dout * gamma
  dmean = -1/np.sqrt(var+eps) * np.sum(dxhat,axis=0)
  dvar = np.sum(-1/2* 1/(var+eps)**1.5 * (x-mean) *dxhat, axis=0)
  dx = 1/np.sqrt(var+eps) * dxhat + 2*(x-mean)/NHW * dvar + 1/NHW*dmean

  # reshape back
  dx = dx.reshape(N, C, H, W)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta