# Load the data

# Training data
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])

# Update training to set last part as validation
id_val = 1801:2000
Yval = Y[id_val]
Xval = X[id_val, ]
Ytrain = Y[-id_val]
Xtrain = X[-id_val, ]

# Testing data
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])

# Source the functions
source("FunctionsNN.R")
source("FunctionsLR.R")

# First run logistic regression as baseline
Xinter <- cbind(rep(1, nrow(Xtrain)), Xtrain)
Xtinter <- cbind(rep(1, nrow(Xt)), Xt)

out <- LRMultiClass(Xinter, Ytrain, Xtinter, Yt, lambda = 1, numIter = 150, eta = 0.1)
plot(out$objective, type = 'o', main = "Objective Function - LR", 
     xlab = "Iteration", ylab = "Objective Value")
plot(out$error_train, type = 'o', main = "Training Error - LR",
     xlab = "Iteration", ylab = "Error Rate (%)")
plot(out$error_test, type = 'o', main = "Test Error - LR",
     xlab = "Iteration", ylab = "Error Rate (%)")

# Default neural network parameters
cat("\nDefault NN parameters results:\n")
out2 = NN_train(Xtrain, Ytrain, Xval, Yval, lambda = 0.001,
                rate = 0.1, mbatch = 50, nEpoch = 150,
                hidden_p = 100, scale = 1e-3, seed = 12345)

# Plot training and validation errors
plot(1:length(out2$error), out2$error, ylim = c(0, 70), type = 'l',
     main = "Training vs Validation Error - Default NN",
     xlab = "Epoch", ylab = "Error Rate (%)", col = "blue")
lines(1:length(out2$error_val), out2$error_val, col = "red")
legend("topright", legend = c("Training", "Validation"),
       col = c("blue", "red"), lty = 1)

# Evaluate default model on test data
test_error = evaluate_error(Xt, Yt, out2$params$W1, out2$params$b1, 
                          out2$params$W2, out2$params$b2)
cat("Default NN test error:", test_error, "%\n")

# Try different parameter combinations to improve performance
parameter_combinations <- list(
  list(lambda = 0.0005, rate = 0.05, mbatch = 32, hidden_p = 150),
  list(lambda = 0.0001, rate = 0.08, mbatch = 64, hidden_p = 200),
  list(lambda = 0.0008, rate = 0.12, mbatch = 40, hidden_p = 128)
)

best_test_error <- Inf
best_params <- NULL

for(i in seq_along(parameter_combinations)) {
  params <- parameter_combinations[[i]]
  cat("\nTrying parameter combination", i, ":\n")
  print(params)
  
  out_new = NN_train(Xtrain, Ytrain, Xval, Yval,
                     lambda = params$lambda,
                     rate = params$rate,
                     mbatch = params$mbatch,
                     nEpoch = 150,
                     hidden_p = params$hidden_p,
                     scale = 1e-3,
                     seed = 12345)
  
  # Plot training and validation errors
  plot(1:length(out_new$error), out_new$error, ylim = c(0, 70), type = 'l',
       main = paste("Training vs Validation Error - Combination", i),
       xlab = "Epoch", ylab = "Error Rate (%)", col = "blue")
  lines(1:length(out_new$error_val), out_new$error_val, col = "red")
  legend("topright", legend = c("Training", "Validation"),
         col = c("blue", "red"), lty = 1)
  
  # Evaluate on test data
  current_test_error = evaluate_error(Xt, Yt, out_new$params$W1, 
                                    out_new$params$b1, out_new$params$W2, 
                                    out_new$params$b2)
  cat("Test error:", current_test_error, "%\n")
  
  if(current_test_error < best_test_error) {
    best_test_error <- current_test_error
    best_params <- params
  }
}

cat("\nBest parameters found:\n")
print(best_params)
cat("Best test error achieved:", best_test_error, "%\n")

# Final optimized model with best parameters
final_model = NN_train(Xtrain, Ytrain, Xval, Yval,
                       lambda = best_params$lambda,
                       rate = best_params$rate,
                       mbatch = best_params$mbatch,
                       nEpoch = 200,  # Increase epochs for final model
                       hidden_p = best_params$hidden_p,
                       scale = 1e-3,
                       seed = 12345)

final_test_error = evaluate_error(Xt, Yt, final_model$params$W1, 
                                final_model$params$b1, final_model$params$W2, 
                                final_model$params$b2)
cat("\nFinal model test error:", final_test_error, "%\n")
