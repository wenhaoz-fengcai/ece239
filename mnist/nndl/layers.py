import numpy as np
import pdb

def affine_forward(x, w, b):

  # print(x)
  x_shape = x.shape

  x = x.reshape((x.shape[0],-1)) #N*D

  # print(x.shape)
  out = x.dot(w)  + b.reshape((1,b.shape[0])) # N*M

  x = x.reshape(x_shape)

  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):

  x, w, b = cache
  dx, dw, db = None, None, None

  x_shape = x.shape
  dw = x.reshape((x.shape[0],-1)).T.dot(dout) # D M
  dx = dout.dot(w.T).reshape(x_shape) # N D

  db = np.sum(dout.T,axis = 1, keepdims = True).T # M*1

  return dx, dw, db

def relu_forward(x):

  out =  np.maximum(0,x)

  cache = x
  return out, cache


def relu_backward(dout, cache):

  x = cache

  dx = (x>0)*(dout)

  return dx

def svm_loss(x, y):

  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):


  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
