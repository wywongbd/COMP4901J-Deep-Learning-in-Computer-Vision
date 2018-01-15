import numpy as np
import math
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  C = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(N):
    x_i = X[i]
    scores = np.dot(x_i, W)
    exp_scores = np.exp(scores)
    loss += (-1) * scores[y[i]] + math.log(np.sum(exp_scores))     
    
    dW[:, y[i]] += (-1) * x_i 
    
    for j in range(C):
        dW[:, j] += (1 / np.sum(exp_scores)) * exp_scores[j] * x_i

  ##############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= N 
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= N
  dW += reg*W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  C = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  XW_product = X.dot(W)
  XW_exp = np.exp(XW_product)
  XW_exp_row_sum = np.sum(XW_exp, axis = 1)
  XW_log_exp_row_sum = np.log(XW_exp_row_sum)
    
  XW_exp = XW_exp.T / XW_exp_row_sum
  XW_exp = XW_exp.T

  binary = np.zeros((N, C))
  binary[np.arange(N), y] = 1
  binary = np.dot(X.T, binary)
    
  dW = (-1) * binary + np.dot(X.T, XW_exp)
  dW /= N
  dW += reg * W

  correct_class_score = XW_product[np.arange(N), y]
  loss += (-1) * np.sum(correct_class_score) + np.sum(XW_log_exp_row_sum)
  loss /= N
  loss += 0.5 * reg * np.sum(W * W)
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

