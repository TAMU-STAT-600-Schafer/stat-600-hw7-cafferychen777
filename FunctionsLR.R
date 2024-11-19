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
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters
  ###################################
  # Check that the first column of X and Xt are 1s
  if(!all(X[,1] == 1) || !all(Xt[,1] == 1)) {
    stop("First column of X and Xt must be 1s for intercept.")
  }
  
  # Dimension checks
  if(nrow(X) != length(y)) {
    stop("Number of rows in X must match length of y.")
  }
  if(nrow(Xt) != length(yt)) {
    stop("Number of rows in Xt must match length of yt.")
  }
  if(ncol(X) != ncol(Xt)) {
    stop("Number of columns in X and Xt must be the same.")
  }
  
  # Check eta and lambda
  if(eta <= 0) {
    stop("Learning rate eta must be positive.")
  }
  if(lambda < 0) {
    stop("Ridge parameter lambda must be non-negative.")
  }
  
  # Determine number of classes K
  K <- length(unique(y))
  
  # Initialize beta
  p <- ncol(X)
  if(is.null(beta_init)) {
    beta <- matrix(0, nrow = p, ncol = K)
  } else {
    if(nrow(beta_init) != p || ncol(beta_init) != K) {
      stop("Dimensions of beta_init are incompatible.")
    }
    beta <- beta_init
  }
  
  ## Helper functions
  ##########################################################################
  # Calculate probabilities
  calculate_probs <- function(X, beta) {
    exp_vals <- exp(X %*% beta)
    probs <- exp_vals / rowSums(exp_vals)
    return(probs)
  }
  
  # Calculate objective value
  calculate_objective <- function(probs, y, beta, lambda) {
    indices <- cbind(seq_len(nrow(probs)), y+1)
    log_likelihood <- sum(log(probs[indices]))
    regularization <- (lambda/2) * sum(beta^2)
    return(-log_likelihood + regularization)
  }
  
  # Calculate classification error
  calculate_error <- function(probs, y) {
    predictions <- max.col(probs) - 1
    error_rate <- mean(predictions != y) * 100
    return(error_rate)
  }
  
  # Initialize vectors to store errors and objective values
  error_train <- numeric(numIter + 1)
  error_test <- numeric(numIter + 1)
  objective <- numeric(numIter + 1)
  
  # Calculate initial probabilities and metrics
  probs_train <- calculate_probs(X, beta)
  error_train[1] <- calculate_error(probs_train, y)
  probs_test <- calculate_probs(Xt, beta)
  error_test[1] <- calculate_error(probs_test, yt)
  objective[1] <- calculate_objective(probs_train, y, beta, lambda)
  
  ## Newton's method iterations
  ##########################################################################
  for(iter in 1:numIter) {
    # Calculate probabilities on training data
    probs <- calculate_probs(X, beta)
    
    # Update beta for each class
    for(k in 1:K) {
      # Calculate W_k diagonals
      w_k <- probs[,k] * (1 - probs[,k])
      
      # Compute Xw efficiently
      Xw <- X * sqrt(w_k)
      
      # Compute Hessian using crossprod
      XtWX <- crossprod(Xw)
      diag(XtWX) <- diag(XtWX) + lambda  # Regularization
      
      # Compute gradient
      residuals_k <- probs[,k] - (y == (k - 1))
      gradient <- t(X) %*% residuals_k + lambda * beta[,k]
      
      # Solve for delta_beta using Cholesky decomposition
      L <- chol(XtWX)
      delta_beta <- backsolve(L, forwardsolve(t(L), gradient))
      
      # Update beta_k
      beta[,k] <- beta[,k] - eta * delta_beta
    }
    
    # Calculate updated probabilities and metrics
    probs_train <- calculate_probs(X, beta)
    probs_test <- calculate_probs(Xt, beta)
    error_train[iter + 1] <- calculate_error(probs_train, y)
    error_test[iter + 1] <- calculate_error(probs_test, yt)
    objective[iter + 1] <- calculate_objective(probs_train, y, beta, lambda)
  }
  
  ## Return output
  ##########################################################################
  return(list(beta = beta, 
              error_train = error_train, 
              error_test = error_test, 
              objective = objective))
}