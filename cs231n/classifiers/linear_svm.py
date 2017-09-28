import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    boolean = (scores - correct_class_score + 1 > 0).astype(int)
    boolean[y[i]] = 0
    
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] -= np.sum(boolean)*(X[i].T)
        continue
      dW[:,j] += boolean[j]*(X[i].T)
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  N = X.shape[0]
  C = W.shape[1]
  D = W.shape[0]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  XW_product = np.dot(X, W)
  label_class_score = XW_product[np.arange(N), y]
  temp = XW_product.T - label_class_score + 1
  temp = temp.T
  temp[np.arange(N), y] = 0
  margin = np.maximum(temp, np.zeros((N, C)))

  loss = np.sum(margin)
  loss /= N
  loss += 0.5 * reg * np.sum(np.multiply(W,W))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
 

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  binary = margin
  binary[binary > 0] = 1
  horizontal_sum = np.sum(binary, axis = 1)
 
  binary[np.arange(N), y] = -horizontal_sum[range(N)]
  
  dW = np.dot(X.T, binary)
  dW /= N
    
  dW += reg*W


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
