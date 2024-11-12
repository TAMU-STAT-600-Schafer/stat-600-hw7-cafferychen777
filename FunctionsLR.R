# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 

## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL) {
  # Get dimensions
  n = nrow(X)
  p = ncol(X)
  K = length(unique(y))
  
  # Check first column is all 1s
  if (!all(X[,1] == 1) || !all(Xt[,1] == 1)) {
    stop("First column of X and Xt must be all 1s")
  }
  
  # Check dimensions
  if (length(y) != n) {
    stop("Incompatible dimensions between X and y")
  }
  if (length(yt) != nrow(Xt)) {
    stop("Incompatible dimensions between Xt and yt")
  }
  if (ncol(X) != ncol(Xt)) {
    stop("Incompatible dimensions between X and Xt")
  }
  
  # Check parameters
  if (eta <= 0) {
    stop("Learning rate eta must be positive")
  }
  if (lambda < 0) {
    stop("Ridge parameter lambda must be non-negative")
  }
  
  # Initialize beta
  if (is.null(beta_init)) {
    beta = matrix(0, nrow = p, ncol = K)
  } else {
    if (nrow(beta_init) != p || ncol(beta_init) != K) {
      stop("Incompatible dimensions for beta_init")
    }
    beta = beta_init
  }
  
  # Initialize storage for errors and objective values
  error_train = numeric(numIter + 1)
  error_test = numeric(numIter + 1)
  objective = numeric(numIter + 1)
  
  # Helper function to compute probabilities
  compute_probs = function(X, beta) {
    scores = X %*% beta
    scores = scores - apply(scores, 1, max) # For numerical stability
    exp_scores = exp(scores)
    probs = exp_scores / rowSums(exp_scores)
    return(probs)
  }
  
  # Helper function to compute objective value
  compute_objective = function(X, y, beta, lambda) {
    probs = compute_probs(X, beta)
    n = nrow(X)
    # Negative log likelihood
    nll = -mean(log(probs[cbind(1:n, y + 1)]))
    # Ridge penalty
    ridge = (lambda/2) * sum(beta^2)
    return(nll + ridge)
  }
  
  # Helper function to compute error rate
  compute_error = function(X, y, beta) {
    probs = compute_probs(X, beta)
    predictions = max.col(probs) - 1
    return(mean(predictions != y) * 100)
  }
  
  # Initial errors and objective
  error_train[1] = compute_error(X, y, beta)
  error_test[1] = compute_error(Xt, yt, beta)
  objective[1] = compute_objective(X, y, beta, lambda)
  
  # Gradient descent iterations
  for (iter in 1:numIter) {
    # Compute probabilities
    probs = compute_probs(X, beta)
    
    # Compute gradient
    grad = matrix(0, nrow = p, ncol = K)
    for (k in 0:(K-1)) {
      indicator = (y == k)
      grad[,k+1] = -colMeans(X * (indicator - probs[,k+1]))
    }
    
    # Add ridge penalty gradient
    grad = grad + lambda * beta
    
    # Update beta
    beta = beta - eta * grad
    
    # Store errors and objective
    error_train[iter + 1] = compute_error(X, y, beta)
    error_test[iter + 1] = compute_error(Xt, yt, beta)
    objective[iter + 1] = compute_objective(X, y, beta, lambda)
  }
  
  return(list(
    beta = beta,
    error_train = error_train,
    error_test = error_test,
    objective = objective
  ))
}