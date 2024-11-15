# Tests.R
# Unit tests for Neural Network implementation

# Source the functions
source("FunctionsNN.R")

# Test 1: Initialize weights and biases
test_initialize <- function() {
  p <- 4
  hidden_p <- 3
  K <- 2
  params <- initialize_bw(p, hidden_p, K, scale = 0.01, seed = 42)
  
  # Check dimensions
  stopifnot(dim(params$W1) == c(p, hidden_p))
  stopifnot(dim(params$W2) == c(hidden_p, K))
  stopifnot(length(params$b1) == hidden_p)
  stopifnot(length(params$b2) == K)
  
  cat("Initialize test passed\n")
}

# Test 2: Loss and gradient computation
test_loss_grad <- function() {
  set.seed(42)
  n <- 5
  K <- 3
  scores <- matrix(rnorm(n*K), n, K)
  y <- sample(0:(K-1), n, replace = TRUE)
  
  result <- loss_grad_scores(y, scores, K)
  
  # Check output structure
  stopifnot(!is.null(result$loss))
  stopifnot(!is.null(result$grad))
  stopifnot(!is.null(result$error))
  
  # Check dimensions
  stopifnot(dim(result$grad) == c(n, K))
  
  cat("Loss and gradient test passed\n")
}

# Test 3: Simple training case
test_simple_training <- function() {
  # Create simple synthetic data
  set.seed(42)
  n <- 100
  p <- 2
  X <- matrix(rnorm(n*p), n, p)
  y <- sample(0:1, n, replace = TRUE)
  
  # Train model
  out <- NN_train(X, y, X[1:10,], y[1:10], 
                  lambda = 0.01, rate = 0.1,
                  mbatch = 10, nEpoch = 5,
                  hidden_p = 3)
  
  # Check output structure
  stopifnot(length(out$error) == 5)
  stopifnot(length(out$error_val) == 5)
  stopifnot(!is.null(out$params$W1))
  
  cat("Simple training test passed\n")
}

# Run all tests
cat("Running tests...\n")
test_initialize()
test_loss_grad()
test_simple_training()
cat("All tests passed!\n")
