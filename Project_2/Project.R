# Case studies WUT 2025
# Mateusz Zacharecki, Patrycja Żak

# Load dataset
setwd('C:/Users/User/Desktop/STUDIA/Rok 2/Warsztaty badawcze/FUZZY/Antonio')
data <- read.table('data_caseStudy.dat', header = TRUE, sep = ',')

# Load required libraries
library(FuzzyNumbers)         # For fuzzy numbers and their operations
library(caret)                # For cross-validation
library(fuzzyreg)             # For possibilistic fuzzy regression
library(FuzzyResampling)      # For Bertoluzza's distance
source("utilities.R")         # Utility functions 
library(ggplot2)              # For plots in point 5

set.seed(935)  # Fix seed for reproducibility

##################### 1. Fuzzification and Defuzzification #####################

n <- nrow(data)  # Number of observations

# Create a list of trapezoidal fuzzy numbers from bounds and mode
intimacy_fuzzy <- list()
for (i in 1:n) {
  intimacy_fuzzy[[i]] <- TrapezoidalFuzzyNumber(
    a1 = data$intimacy_lb[i],       # Lower bound
    a2 = data$intimacy_m[i],        # Mode 
    a3 = data$intimacy_m[i],        # Mode 
    a4 = data$intimacy_ub[i]        # Upper bound
  )
}

# Compute expected value (Delgado's defuzzification)
intimacy_ev <-sapply(intimacy_fuzzy, expectedValue)
intimacy_defuzz <- unlist(lapply(intimacy_fuzzy, expectedValue))

# Compute ambiguity (Delgado's fuzziness measure)
intimacy_ambiguity <- sapply(intimacy_fuzzy, ambiguity)

# Plot defuzzified values
hist(intimacy_defuzz,
     main = "Delgado's Expected Value of Intimacy",
     xlab = "Defuzzified Intimacy",
     col = "lightgreen",
     border = "darkgreen")

# Plot fuzziness (ambiguity)
hist(intimacy_ambiguity,
     main = "Delgado's Ambiguity of Intimacy",
     xlab = "Fuzziness",
     col = "lightgreen",
     border = "darkgreen")

##################### 2. Prepare dataset for regression #####################

# Convert partner gender to numeric factor: 0 = female, 1 = male
data$Gender_of_partner <- ifelse(data$Gender_of_partner == "female", 0,
                                 ifelse(data$Gender_of_partner == "male", 1, NA))

# Create final dataset for regression models
data_reg <- data.frame(
  age = data$Age,
  rel_length = data$Rel_length,
  desire = data$desire,
  gender_of_partner = as.factor(data$Gender_of_partner),
  partner_respo = data$partner_respo,
  intimacy_m = data$intimacy_m,                       # Mode of intimacy
  y.left = data$intimacy_m - data$intimacy_lb,        # Left spread
  y.right = data$intimacy_ub - data$intimacy_m,       # Right spread
  intimacy_ev = intimacy_ev,                          # EV from 1a
  intimacy_ambiguity = intimacy_ambiguity             # Ambiguity
)

##################### 3. K-fold Cross-Validation Setup #####################

K <- 5  # Number of folds
folds <- createFolds(data_reg$intimacy_ev, k = K, returnTrain = FALSE)

# Function needed later in loop for Bertoluzza function
fuzzy_list_to_matrix <- function(fuzzy_list) {
  do.call(rbind, lapply(fuzzy_list, function(f) {
    c(f@a1, f@a2, f@a3, f@a4)
  }))
}

# Initialize storage for prediction errors
init_metrics <- function() list(lm_a = c(), lm_b = c(), plr = c(), flr_interactive = c(), fuzzy_mle = c())
rmse_results <- init_metrics()
mae_results <- init_metrics()
bertoluzza_results <- init_metrics()

# ##################### 4. Main Cross-Validation Loop #####################

for (k in 1:K) {
  cat(sprintf("Running fold %d/%d\n", k, K))  # Show current fold number
  
  # Split the data into training and testing sets
  test_idx <- folds[[k]]
  train_data <- data_reg[-test_idx, ]
  test_data <- data_reg[test_idx, ]
  intimacy_fuzzy_test <- intimacy_fuzzy[test_idx]  # True fuzzy values for test set
  
  # Define predictor formula and compute model matrix for test set
  predictors_formula <- ~ age + rel_length + desire + gender_of_partner + partner_respo
  X_test <- model.matrix(predictors_formula, data = test_data)
  
  ######### (a) Linear Regression #########
  model_lm_a <- lm(intimacy_ev ~ age + rel_length + desire + gender_of_partner + partner_respo, 
                 data = train_data)
  # Fit standard linear model
  pred_lm_a <- predict(model_lm_a, newdata = test_data)  # Predict on test set
  
  ######### (b) Weighted Linear Regression #########
  model_lm_b <- lm(intimacy_ev ~ age + rel_length + desire + gender_of_partner + partner_respo, 
                        data = train_data,
                        weights = train_data$intimacy_ambiguity)
  pred_lm_b <- predict(model_lm_b, newdata = test_data)
  
  ######### (c) Possibilistic Linear Regression (PLR) #########
  model_plr <- fuzzylm(formula = intimacy_m ~ age + rel_length + desire + gender_of_partner + partner_respo,
                       data = train_data,
                       fuzzy.left.y = "y.left", fuzzy.right.y = "y.right",
                       method = "plr")  # Fit PLR model
  coefs <- coef(model_plr)
  
  # Generate fuzzy predictions using PLR model coefficients
  
  beta_m  <- coef(model_plr)[, 1]  # mode
  beta_l  <- coef(model_plr)[, 2]  # left spread
  beta_r  <- coef(model_plr)[, 3]  # right spread
  
  modc_m  <- X_test %*% beta_m
  modc_lb <- modc_m - X_test %*% beta_l
  modc_ub <- modc_m + X_test %*% beta_r
  
  pred_plr_fuzzy <- vector("list", nrow(test_data))
  for(i in seq_len(nrow(test_data))) {
    pred_plr_fuzzy[[i]] <- TrapezoidalFuzzyNumber(
      a1 = modc_lb[i],  # left bound
      a2 = modc_m[i],   # mode 
      a3 = modc_m[i],   # mode 
      a4 = modc_ub[i]   # right bound
    )
  }
  
  pred_plr_defuzz <- sapply(pred_plr_fuzzy, expectedValue)
  
  ######### (d) Fuzzy Least Squares (Interactive) #########
  X_train_flr <- model.matrix(model_lm_a)
  J_flr <- ncol(X_train_flr) - 1 # Number of predictors
  # Estimate parameters via optimization (interactive model)
  res_flr <- optim(
    fn = flr1,
    par = rep(1, J_flr + 3),  # (J+1 betas, 1 left_ratio, 1 right_ratio)
    method = "L-BFGS-B",
    lower = c(rep(-Inf, J_flr + 1), 1e-4, 1e-4),
    upper = c(rep(Inf, J_flr + 1), Inf, Inf),
    m = train_data$intimacy_m,
    l = train_data$y.left,
    r = train_data$y.right,
    X = X_train_flr,
    J = J_flr,
    control = list(maxit = 10000, trace = 1)
  )

  # Extract optimized parameters
  beta_flr_m <- res_flr$par[1:(J_flr + 1)]  # coefs for means
  beta_flr_l <- res_flr$par[J_flr + 2]      # left coef (ratio)
  beta_flr_r <- res_flr$par[J_flr + 3]      # right coef (spread ratio)
  
  modd_m <- X_test %*% beta_flr_m
  modd_ls <- modd_m * beta_flr_l
  modd_rs <- modd_m * beta_flr_r
  
  pred_flr_fuzzy <- list()
  for(i in (1:nrow(X_test))) {
    pred_flr_fuzzy[[i]] <- TrapezoidalFuzzyNumber(
      a1 = modd_m[i] - modd_ls[i],
      a2 = modd_m[i],
      a3 = modd_m[i],
      a4 = modd_m[i] + modd_rs[i]
    )
  }

  # Expected value
  pred_flr_defuzz <- sapply(pred_flr_fuzzy, expectedValue)
  ######### (e) Fuzzy MLE #########
  X_train_mle <- model.matrix(model_lm_a)
  
  # Optimize log-likelihood of fuzzy regression model
  J <- ncol(X_train_mle)
  
  start_beta <- coef(model_lm_a)
  start_par <- c(start_beta, 1)
  res_mle <- optim(par = start_par,
                   fn = likelFun,
                   method = "L-BFGS-B",
                   lower = c(rep(-Inf, J), 1e-6),  # sigma > 0
                   m1 = train_data$intimacy_m,
                   m2 = train_data$intimacy_m,
                   l = train_data$y.left,
                   r = train_data$y.right,
                   X = X_train_mle)
  
  beta_mle <- res_mle$par[1:J]  # Extract coefficients
  pred_mle <- X_test %*% beta_mle  # Make predictions
  pred_mle_defuzz <- as.vector(pred_mle)  # Convert to numeric vector
  
  # Represent as degenerate fuzzy numbers
  pred_mle_fuzzy <- lapply(pred_mle_defuzz, 
                           function(val) TrapezoidalFuzzyNumber(val, val, val, val))
  
  ######### Evaluation: RMSE, MAE #########
  y_true <- test_data$intimacy_ev
  rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))  # Root Mean Squared Error
  mae <- function(y, yhat) mean(abs(y - yhat))        # Mean Absolute Error
  
  # Store errors for each model
  rmse_results$lm_a <- c(rmse_results$lm_a, rmse(y_true, pred_lm_a))
  mae_results$lm_a <- c(mae_results$lm_a, mae(y_true, pred_lm_a))
  
  rmse_results$lm_b <- c(rmse_results$lm_b, rmse(y_true, pred_lm_b))
  mae_results$lm_b <- c(mae_results$lm_b, mae(y_true, pred_lm_b))
  
  rmse_results$plr <- c(rmse_results$plr, rmse(y_true, pred_plr_defuzz))
  mae_results$plr <- c(mae_results$plr, mae(y_true, pred_plr_defuzz))
  
  rmse_results$flr_interactive <- c(rmse_results$flr_interactive, rmse(y_true, pred_flr_defuzz))
  mae_results$flr_interactive <- c(mae_results$flr_interactive, mae(y_true, pred_flr_defuzz))
  
  rmse_results$fuzzy_mle <- c(rmse_results$fuzzy_mle, rmse(y_true, pred_mle_defuzz))
  mae_results$fuzzy_mle <- c(mae_results$fuzzy_mle, mae(y_true, pred_mle_defuzz))
  
  ######### Bertoluzza's distance #########
  # Bertoluzza's distance computed only for models with fuzzy predictions
  # Default parameter: 1/3
  
  true_matrix <- fuzzy_list_to_matrix(intimacy_fuzzy_test)
  pred_matrix_plr <- fuzzy_list_to_matrix(pred_plr_fuzzy)
  pred_matrix_flr <- fuzzy_list_to_matrix(pred_flr_fuzzy)
  pred_matrix_mle <- fuzzy_list_to_matrix(pred_mle_fuzzy)
  
  bertoluzza_results$plr <- c(bertoluzza_results$plr,
                              mean(BertoluzzaDistance(true_matrix, pred_matrix_plr)))
  
  bertoluzza_results$flr_interactive <- c(bertoluzza_results$flr_interactive,
                                          mean(BertoluzzaDistance(true_matrix, pred_matrix_flr)))
  
  bertoluzza_results$fuzzy_mle <- c(bertoluzza_results$fuzzy_mle,
                                    mean(BertoluzzaDistance(true_matrix, pred_matrix_mle)))
}

##################### 5. Results #####################
summary_df <- data.frame(
  Model = names(rmse_results),
  RMSE_mean = sapply(rmse_results, mean),
  RMSE_sd   = sapply(rmse_results, sd),
  MAE_mean  = sapply(mae_results, mean),
  MAE_sd    = sapply(mae_results, sd),
  Bertoluzza_mean = sapply(bertoluzza_results, mean),
  Bertoluzza_sd   = sapply(bertoluzza_results, sd)
)
print(summary_df)

# Model            RMSE_mean  RMSE_sd     MAE_mean   MAE_sd     Bertoluzza_mean Bertoluzza_sd
# lm_a             0.3577101  0.01722359  0.2658966  0.01234348 NA              NA
# lm_b             0.3704111  0.01789652  0.2835409  0.01118190 NA              NA
# plr              0.4865370  0.04625957  0.3679057  0.03480049 1.2231788       0.02220288
# flr_interactive  0.3532938  0.01394165  0.2678034  0.01186670 0.2797195       0.01141906
# fuzzy_mle        37.0049522 0.86635954  35.5894769 0.70836068 35.5898990      0.70835918

# RMSE plot
rmse_df <- data.frame(
  Model = rep(names(rmse_results), each = K),
  RMSE = unlist(rmse_results)
)

ggplot(rmse_df, aes(x = Model, y = RMSE)) +
  geom_boxplot(fill = "lightblue") +
  ggtitle("RMSE across models") +
  theme_minimal()

# Bertoluzza plot
bert_df <- data.frame(
  Model = rep(names(bertoluzza_results), each = K)[11:25],
  Distance = unlist(bertoluzza_results)
)

ggplot(bert_df, aes(x = Model, y = Distance)) +
  geom_boxplot(fill = "lightgreen") +
  ggtitle("Bertoluzza distance across models") +
  theme_minimal()

# Answear

# In general:
## Based on RMSE and MAE, the lowest prediction error is achieved by lm_a (standard linear regression).
## Among the fuzzy models, the flr_interactive model performs best, with comparable RMSE/MAE to lm_a, and the lowest Bertoluzza distance, indicating the best fuzzy shape preservation.

# Based on tructural similarities:

## 1. lm_a - lm_b
### These are structurally identical models, differing only by weighting. 

### lm_a slightly outperforms lm_b, suggesting that weighting by ambiguity may 
### not improve accuracy—and may even add noise.

## 2. flr_interactive - plr
### Both are fuzzy models, but plr is possibilistic and fits mode + spreads 
### directly; flr_interactive is an interactive fuzzy least squares approach.

### flr_interactive clearly dominates: lower RMSE, MAE, and Bertoluzza distance.

### Interpretation: accounting for the spread dynamically via optimization (as in FLR) is more robust than fixed structure of PLR.

## 3. fuzzy_mle
### Performs worst by far, with huge errors (RMSE ~ 37).
### Likely explanation:
### - Optimization does not converge properly.
### - The log-likelihood function is numerically unstable, possibly due to divergent integrals or flat gradients.
### - Despite adjustments, the fitted model produces degenerate fuzzy outputs far from observed values.

# Conclusions:
# 1. lm_a is the best overall in terms of defuzzified accuracy.
# 2. However, flr_interactive provides the best fuzzy predictions, preserving uncertainty structure with low Bertoluzza distance.
# 3. fuzzy_mle is not viable without significant improvements to numerical stability or better initialization.