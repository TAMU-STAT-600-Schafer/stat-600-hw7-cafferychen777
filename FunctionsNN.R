# Neural Network Implementation for Multi-class Classification
#####################################################
# This file implements a two-layer neural network with ReLU activation
# for multi-class classification problems.

# Initialize weights and biases for neural network
#####################################################
# Purpose: Initialize weights and biases for a 2-layer neural network
# 
# Parameters:
#   p: Integer - Input layer dimension (number of features)
#   hidden_p: Integer - Hidden layer dimension (number of neurons)
#   K: Integer - Output layer dimension (number of classes)
#   scale: Float - Standard deviation for weight initialization (default: 1e-3)
#   seed: Integer - Random seed for reproducibility (default: 12345)
#
# Returns:
#   List containing:
#     b1: Vector of length hidden_p - First layer biases
#     b2: Vector of length K - Second layer biases
#     W1: Matrix of dim p x hidden_p - First layer weights
#     W2: Matrix of dim hidden_p x K - Second layer weights
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  # Print initialization parameters for debugging
  cat("\nStarting initialize_bw function\n")
  cat("Parameters:\n")
  cat("p:", p, "\n")
  cat("hidden_p:", hidden_p, "\n")
  cat("K:", K, "\n")
  cat("scale:", scale, "\n")
  
  # Set random seed for reproducibility
  set.seed(seed)
  
  # Initialize biases as zeros (common practice in neural networks)
  b1 = rep(0, hidden_p)  # First layer bias vector
  b2 = rep(0, K)         # Second layer bias vector
  
  cat("\nInitializing weights\n")
  # Initialize weights using small random values from normal distribution
  # Small initialization helps prevent saturation of neurons
  W1 = matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p)
  W2 = matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p)
  
  # Verify dimensions of initialized parameters
  cat("\nDimensions check:\n")
  cat("W1:", dim(W1), "\n")
  cat("W2:", dim(W2), "\n")
  cat("b1 length:", length(b1), "\n")
  cat("b2 length:", length(b2), "\n")
  
  # Print sample values for verification
  cat("\nFirst few values of W1:\n")
  print(W1[1:2, 1:2])
  cat("\nFirst few values of W2:\n")
  print(W2[1:2, 1:2])
  
  cat("\nFinished initialize_bw function\n")
  
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Calculate loss, error, and gradient based on scores
#############################################################
# Purpose: Compute cross-entropy loss, gradients, and classification error
#          for multi-class classification using softmax
#
# Parameters:
#   scores - n x K matrix of scores (output layer values before softmax)
#   y - n-dimensional vector of class labels (0 to K-1)
#   K - number of classes
#
# Technical Details:
#   1. Softmax computation includes numerical stability adjustment
#   2. Cross-entropy loss is averaged over all samples
#   3. Gradients are computed for backpropagation
#   4. Error rate is percentage of misclassifications
#
# Returns: List containing
#   - loss: Float - Cross-entropy loss (averaged over samples)
#   - grad: Matrix - Gradients with respect to scores (n x K)
#   - error: Float - Classification error rate (%)
loss_grad_scores <- function(y, scores, K){
  n = length(y)
  
  # Compute softmax probabilities with numerical stability
  # Subtract max score to prevent overflow in exp()
  scores_shifted = scores - apply(scores, 1, max)
  exp_scores = exp(scores_shifted)
  probs = exp_scores / rowSums(exp_scores)
  
  # Compute cross-entropy loss
  # Add 1 to y for R's 1-based indexing
  correct_logprobs = -log(probs[cbind(1:n, y + 1)])
  loss = mean(correct_logprobs)
  
  # Compute gradient for backpropagation
  # grad = softmax - one_hot_encoded_labels
  grad = probs
  grad[cbind(1:n, y + 1)] = grad[cbind(1:n, y + 1)] - 1
  grad = grad / n  # Average over samples
  
  # Compute classification error rate
  predictions = max.col(scores) - 1  # Convert to 0-based class labels
  error = mean(predictions != y) * 100
  
  return(list(loss = loss, grad = grad, error = error))
}

# Forward and backward pass through the network
################################################
# Parameters:
# X - n x p matrix of input features
# y - n-dimensional vector of class labels (0 to K-1)
# W1 - p x hidden_p matrix of first layer weights
# b1 - hidden_p-dimensional vector of first layer biases
# W2 - hidden_p x K matrix of second layer weights
# b2 - K-dimensional vector of second layer biases
# lambda - L2 regularization parameter
#
# Returns:
# - loss: total loss (cross-entropy + regularization)
# - error: classification error rate (%)
# - grads: list containing gradients for W1, b1, W2, b2
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda) {
  # Parameters:
  #   X: Input data matrix (n_samples x n_features)
  #   y: Target labels (0-based indexing)
  #   K: Number of classes
  #   W1, b1: First layer weights and biases
  #   W2, b2: Second layer weights and biases
  #   lambda: Regularization strength
  
  # Forward pass
  # ------------
  # Step 1: Input to hidden layer with ReLU activation
  hidden = X %*% W1
  hidden = sweep(hidden, 2, b1, "+")
  hidden_relu = matrix(pmax(0, hidden), nrow = nrow(hidden))
  
  # Step 2: Hidden to output layer
  scores = hidden_relu %*% W2
  scores = sweep(scores, 2, b2, "+")
  
  # Compute loss and gradients
  # Compute softmax probabilities
  scores_exp = exp(scores)
  probs = sweep(scores_exp, 1, rowSums(scores_exp), "/")
  
  # Calculate loss: cross-entropy loss + L2 regularization
  correct_logprobs = -log(probs[cbind(1:nrow(probs), y + 1)])
  data_loss = mean(correct_logprobs)
  reg_loss = 0.5 * lambda * (sum(W1 * W1) + sum(W2 * W2))
  loss = data_loss + reg_loss
  
  # Calculate error
  predictions = max.col(probs) - 1
  error = mean(predictions != y) * 100
  
  # Backward pass
  dscores = probs
  dscores[cbind(1:nrow(dscores), y + 1)] = dscores[cbind(1:nrow(dscores), y + 1)] - 1
  dscores = dscores / nrow(X)
  
  # Gradients for W2 and b2
  dW2 = t(hidden_relu) %*% dscores
  db2 = colSums(dscores)
  
  # Gradients for hidden layer
  dhidden = dscores %*% t(W2)
  dhidden[hidden <= 0] = 0
  
  # Gradients for W1 and b1
  dW1 = t(X) %*% dhidden
  db1 = colSums(dhidden)
  
  # Add regularization gradient
  dW1 = dW1 + lambda * W1
  dW2 = dW2 + lambda * W2
  
  return(list(loss = loss, error = error, 
              grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)))
}

# Evaluate model performance on validation set
####################################################
# Purpose: Calculate classification error rate on validation dataset using
#          the trained neural network model parameters
#
# Parameters:
#   Xval - nval x p matrix of validation features
#          nval: number of validation samples
#          p: number of input features
#   yval - nval-dimensional vector of validation labels (0 to K-1)
#          K: number of classes
#   W1 - p x hidden_p matrix of first layer weights
#   b1 - hidden_p-dimensional vector of first layer biases
#   W2 - hidden_p x K matrix of second layer weights
#   b2 - K-dimensional vector of second layer biases
#
# Technical Details:
#   1. Performs forward pass through the network:
#      - First layer with ReLU activation
#      - Second layer producing final scores
#   2. Uses matrix operations for efficient computation
#   3. Converts scores to predictions using argmax
#   4. Calculates error rate as percentage of misclassifications
#
# Returns:
#   error - Float, classification error rate (%) on validation set
#           calculated as: (number of incorrect predictions / total samples) * 100
#
# Example:
#   val_error <- evaluate_error(X_validation, y_validation, W1, b1, W2, b2)
#   print(paste("Validation error:", val_error, "%"))
evaluate_error <- function(Xval, yval, W1, b1, W2, b2) {
  # Forward pass through the network
  # -------------------------------
  
  # First layer
  hidden = Xval %*% W1
  hidden = sweep(hidden, 2, b1, "+")
  hidden_relu = matrix(pmax(0, hidden), nrow = nrow(hidden))
  
  # Second layer
  scores = hidden_relu %*% W2
  scores = sweep(scores, 2, b2, "+")
  
  # Efficient matrix operations for large-scale data
  predictions = max.col(scores) - 1
  error = mean(predictions != yval) * 100
  
  return(error)
}

# Train neural network using mini-batch SGD
################################################
# Purpose: Train a two-layer neural network for multi-class classification
#          using mini-batch stochastic gradient descent
#
# Parameters:
#   X - n x p matrix of training features
#       n: number of training samples
#       p: number of input features
#   y - n-dimensional vector of training labels (0 to K-1)
#       K: number of classes
#   Xval - nval x p matrix of validation features
#   yval - nval-dimensional vector of validation labels
#   lambda - L2 regularization parameter (default: 0.01)
#           Controls the strength of weight regularization
#   rate - learning rate for SGD (default: 0.01)
#          Controls the step size during optimization
#   mbatch - mini-batch size (default: 20)
#            Number of samples per gradient update
#   nEpoch - number of training epochs (default: 100)
#            One epoch is a complete pass through the training data
#   hidden_p - number of hidden layer neurons (default: 20)
#              Controls model capacity
#   scale - weight initialization scale (default: 1e-3)
#           Standard deviation for random weight initialization
#   seed - random seed for reproducibility (default: 12345)
#
# Technical Details:
#   1. Initialization:
#      - Weights are initialized using normal distribution
#      - Biases are initialized to zero
#   2. Training Process:
#      - Data is randomly shuffled into mini-batches each epoch
#      - Forward pass computes predictions
#      - Backward pass computes gradients
#      - Parameters are updated using SGD
#   3. Monitoring:
#      - Training error is averaged across batches
#      - Validation error is computed after each epoch
#
# Returns: List containing
#   - error: Vector of length nEpoch with training errors
#   - error_val: Vector of length nEpoch with validation errors
#   - params: List of final model parameters
#     * W1: First layer weights
#     * b1: First layer biases
#     * W2: Second layer weights
#     * b2: Second layer biases
#
# Example:
#   model <- NN_train(X_train, y_train, X_val, y_val,
#                     lambda = 0.01, rate = 0.01,
#                     mbatch = 32, nEpoch = 100)
#   plot(1:100, model$error, type = "l",
#        xlab = "Epoch", ylab = "Error Rate")
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345){
  n = length(y)
  nBatch = floor(n/mbatch)
  p = ncol(X)
  K = length(unique(y))
  
  # Initialize parameters
  params = initialize_bw(p, hidden_p, K, scale, seed)
  W1 = params$W1
  b1 = params$b1
  W2 = params$W2
  b2 = params$b2
  
  error = rep(NA, nEpoch)
  error_val = rep(NA, nEpoch)
  
  # 确保 Xval 保持矩阵形式
  Xval = as.matrix(Xval)
  
  set.seed(seed)
  for (i in 1:nEpoch){
    # Shuffle data into batches
    batchids = sample(rep(1:nBatch, length.out = n), size = n)
    batch_errors = numeric(nBatch)
    
    # Process each batch
    for(j in 1:nBatch){
      batch_idx = which(batchids == j)
      X_batch = X[batch_idx, , drop = FALSE]  # 防止降维
      y_batch = y[batch_idx]
      
      # Forward and backward pass
      out = one_pass(X_batch, y_batch, K, W1, b1, W2, b2, lambda)
      
      # Update parameters
      W1 = W1 - rate * out$grads$dW1
      b1 = b1 - rate * out$grads$db1
      W2 = W2 - rate * out$grads$dW2
      b2 = b2 - rate * out$grads$db2
      
      batch_errors[j] = out$error
    }
    
    # Record errors
    error[i] = mean(batch_errors)
    error_val[i] = evaluate_error(Xval, yval, W1, b1, W2, b2)
  }
  
  return(list(error = error, error_val = error_val, 
              params = list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}