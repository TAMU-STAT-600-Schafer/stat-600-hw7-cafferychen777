# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  cat("\nStarting initialize_bw function\n")
  cat("Parameters:\n")
  cat("p:", p, "\n")
  cat("hidden_p:", hidden_p, "\n")
  cat("K:", K, "\n")
  cat("scale:", scale, "\n")
  
  set.seed(seed)
  
  # Initialize intercepts as zeros
  b1 = rep(0, hidden_p)
  b2 = rep(0, K)
  
  cat("\nInitializing weights\n")
  # Initialize weights
  W1 = matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p)
  W2 = matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p)
  
  cat("\nDimensions check:\n")
  cat("W1:", dim(W1), "\n")
  cat("W2:", dim(W2), "\n")
  cat("b1 length:", length(b1), "\n")
  cat("b2 length:", length(b2), "\n")
  
  # Verify matrix structure
  cat("\nFirst few values of W1:\n")
  print(W1[1:2, 1:2])
  cat("\nFirst few values of W2:\n")
  print(W2[1:2, 1:2])
  
  cat("\nFinished initialize_bw function\n")
  
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  n = length(y)
  
  # Compute softmax probabilities
  scores_shifted = scores - apply(scores, 1, max) # For numerical stability
  exp_scores = exp(scores_shifted)
  probs = exp_scores / rowSums(exp_scores)
  
  # Compute loss
  correct_logprobs = -log(probs[cbind(1:n, y + 1)])  # +1 for R indexing
  loss = mean(correct_logprobs)
  
  # Compute gradient
  grad = probs
  grad[cbind(1:n, y + 1)] = grad[cbind(1:n, y + 1)] - 1
  grad = grad / n
  
  # Compute error rate
  predictions = max.col(scores) - 1  # -1 to match 0-based class labels
  error = mean(predictions != y) * 100
  
  return(list(loss = loss, grad = grad, error = error))
}

# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
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

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
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
  
  # Compute predictions
  predictions = max.col(scores) - 1
  error = mean(predictions != yval) * 100
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
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