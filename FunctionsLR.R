# Multi-class Logistic Regression implementation using gradient descent
#############################################################
# Input Parameters:
# X - n x p training data matrix, where:
#     - n is the number of observations
#     - p is the number of features (including intercept)
#     - First column must be 1s for intercept term
# y - n-dimensional vector of class labels (0 to K-1)
# Xt - ntest x p testing data matrix (same format as X)
# yt - ntest-dimensional vector of test class labels (0 to K-1)
# numIter - number of gradient descent iterations (default: 50)
# eta - learning rate for gradient descent (default: 0.1)
# lambda - L2 regularization parameter (default: 1)
# beta_init - optional initial coefficient matrix (p x K)

# Return Values:
# beta - p x K matrix of fitted coefficients
# error_train - vector of training error rates (%) for each iteration
# error_test - vector of testing error rates (%) for each iteration
# objective - vector of objective function values (NLL + ridge penalty)

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
  
  # Helper function to compute class probabilities using softmax
  compute_probs = function(X, beta) {
    scores = X %*% beta
    scores = scores - apply(scores, 1, max)  # Subtract max for numerical stability
    exp_scores = exp(scores)
    probs = exp_scores / rowSums(exp_scores)  # Softmax transformation
    return(probs)
  }
  
  # Helper function to compute objective value (NLL + ridge penalty)
  compute_objective = function(X, y, beta, lambda) {
    probs = compute_probs(X, beta)
    n = nrow(X)
    # Negative log likelihood of the correct classes
    nll = -mean(log(probs[cbind(1:n, y + 1)]))
    # L2 regularization term
    ridge = (lambda/2) * sum(beta^2)
    return(nll + ridge)
  }
  
  # Helper function to compute classification error rate (%)
  compute_error = function(X, y, beta) {
    probs = compute_probs(X, beta)
    predictions = max.col(probs) - 1  # Get class with highest probability
    return(mean(predictions != y) * 100)
  }
  
  # Initial errors and objective
  error_train[1] = compute_error(X, y, beta)
  error_test[1] = compute_error(Xt, yt, beta)
  objective[1] = compute_objective(X, y, beta, lambda)
  
  # Main gradient descent loop
  for (iter in 1:numIter) {
    # Step 1: Compute current probabilities
    probs = compute_probs(X, beta)
    
    # Step 2: Compute gradient of negative log likelihood
    grad = matrix(0, nrow = p, ncol = K)
    for (k in 0:(K-1)) {
      indicator = (y == k)
      grad[,k+1] = -colMeans(X * (indicator - probs[,k+1]))
    }
    
    # Step 3: Add gradient of ridge penalty
    grad = grad + lambda * beta
    
    # Step 4: Update coefficients using gradient descent
    beta = beta - eta * grad
    
    # Step 5: Store metrics for current iteration
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