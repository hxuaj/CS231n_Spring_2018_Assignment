import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(num_train):
    score = np.exp(np.dot(X[i],W))   # shape = 1 x C
    right_score = score[y[i]]
    l_i = right_score / np.sum(score)
    l_i = -np.log(l_i)
    for j in range(num_class):
      l_j = score[j] / np.sum(score)
      if j == y[i]:
        dW[:,j] += X[i] * (-1 + l_j)
      else:
        dW[:,j] += X[i] * l_j

    loss += l_i
  loss = loss/num_train + reg * np.sum(W * W)
  dW = dW/num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]

  #Compute the loss
  score = np.exp(np.dot(X,W))
  right_score = score[range(num_train), y]
  score_sum = np.sum(score, axis=1)
  loss = np.sum(-np.log(right_score / score_sum))
  loss = loss/num_train + reg * np.sum(W * W)

  #Compute the gradient
  mask = score / np.reshape(score_sum,(-1,1))
  mask[range(num_train), y] -= 1 
  dW = np.dot(X.T, mask)
  dW = dW/num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

