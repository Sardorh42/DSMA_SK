# ============================================================
# STEP 7 (BALANCED): EFFICIENT HYPERPARAMETER TUNING
# Strategy: 3-Fold CV + "Sweet Spot" Grids
# ============================================================

library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(nnet)
library(adabag)
library(pROC)
library(dplyr)

print("--- Starting Hyperparameter Tuning ---")

# 1. SETUP & HELPER FUNCTIONS
# ------------------------------------------------------------
# 3-Fold CV is efficient and sufficient (presumably)
fitControl <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary # Maximizes ROC/AUC
)

# Helper function to calculate Gini, TDL, and Accuracy
calc_metrics_tuned <- function(actual, predicted_prob) {
  actual_num <- ifelse(actual == "Positive", 1, 0)
  
  # Accuracy (Threshold 0.5)
  pred_class <- ifelse(predicted_prob > 0.5, 1, 0)
  acc <- mean(pred_class == actual_num)
  
  # Top Decile Lift (TDL)
  dat <- data.frame(actual = actual_num, prob = predicted_prob)
  dat <- dat[order(-dat$prob), ]
  top_10 <- floor(0.1 * nrow(dat))
  if(top_10 < 1) top_10 <- 1
  tdl <- mean(dat$actual[1:top_10]) / mean(dat$actual)
  
  # Gini
  roc_obj <- suppressMessages(roc(actual_num, predicted_prob))
  gini <- 2 * as.numeric(auc(roc_obj)) - 1
  
  return(c(Accuracy = acc, TDL = tdl, Gini = gini))
}

# Initialize Results Table
tuned_results <- data.frame(Model=character(), Best_Param=character(), 
                            Accuracy=numeric(), TDL=numeric(), Gini=numeric(), 
                            Time_Sec=numeric(), stringsAsFactors=FALSE)

# 2. TUNING LOOP (CARET MODELS)
# ------------------------------------------------------------
# Use optimal ranges rather than hard limits.

caret_models <- list(
  # Logit: Elastic Net is better than standard Logit
  "Logit" = list(
    method = "glmnet", 
    grid = expand.grid(alpha = c(0, 1), lambda = c(0.001, 0.01))
  ),
  
  # Naive Bayes: fL=1 (Laplace) is NECESSARY to prevent zero-prob errors
  "Naive Bayes" = list(
    method = "nb", 
    grid = expand.grid(fL = c(1, 2), usekernel = c(TRUE, FALSE), adjust = 1)
  ),
  
  # KNN: Higher k reduces noise
  "KNN" = list(
    method = "knn", 
    grid = expand.grid(k = c(15, 35, 55))
  ),
  
  # SVM: Cost function tuning
  "SVM" = list(
    method = "svmRadial", 
    grid = expand.grid(C = c(1, 10), sigma = 0.05)
  ),
  
  # Decision Tree: Needs 'cp' to control depth
  "Decision Tree" = list(
    method = "rpart", 
    grid = expand.grid(cp = c(0.01, 0.001)) # 0.001 allows deep trees
  ),
  
  # Random Forest: 100 trees is optimal balance for tuning (presumably)
  "Random Forest" = list(
    method = "rf", 
    grid = expand.grid(mtry = c(2, 4, 6)), 
    ntree = 100 
  ),
  
  # Neural Net: Needs size (neurons) and decay (regularization)
  "Neural Net" = list(
    method = "nnet", 
    grid = expand.grid(size = c(3, 5), decay = 0.1),
    maxit = 200
  )
)

for(mod_name in names(caret_models)) {
  print(paste("Tuning:", mod_name, "..."))
  start_time <- Sys.time()
  config <- caret_models[[mod_name]]
  
  tryCatch({
    set.seed(123)
    
    # Handle specific model arguments
    if(mod_name == "Random Forest") {
      fit <- train(rating_binary ~ ., data = df_train_scaled, method = config$method,
                   trControl = fitControl, tuneGrid = config$grid, metric = "ROC", 
                   ntree = 100)
    } else if(mod_name == "Neural Net") {
      fit <- train(rating_binary ~ ., data = df_train_scaled, method = config$method,
                   trControl = fitControl, tuneGrid = config$grid, metric = "ROC", 
                   trace = FALSE, maxit = 200)
    } else {
      fit <- train(rating_binary ~ ., data = df_train_scaled, method = config$method,
                   trControl = fitControl, tuneGrid = config$grid, metric = "ROC")
    }
    
    # Predict & Evaluate
    probs <- predict(fit, newdata = df_test_scaled, type = "prob")[, "Positive"]
    metrics <- calc_metrics_tuned(df_test_scaled$rating_binary, probs)
    
    duration <- round(as.numeric(difftime(Sys.time(), start_time, units="secs")), 2)
    best_p <- paste(names(fit$bestTune), fit$bestTune, sep="=", collapse=", ")
    
    tuned_results[nrow(tuned_results)+1, ] <- c(mod_name, best_p, round(metrics, 3), duration)
    print(paste("Done:", mod_name, "| Gini:", round(metrics["Gini"], 3)))
    
    rm(fit); gc() 
    
  }, error = function(e) { print(paste("Error:", e$message)) })
}

# 3. MANUAL TUNING: BAGGING & BOOSTING (Ensembles)
# ------------------------------------------------------------

ada_models <- c("Bagging", "Boosting")
mfinal_grid <- c(10, 50, 100, 200) 

# Bagging needs high variance trees 
complex_control <- rpart.control(maxdepth = 15, cp = 0.001)

for(mod_name in ada_models) {
  print(paste("Tuning:", mod_name, "..."))
  start_time <- Sys.time()
  
  best_gini <- -1
  best_m <- NA
  best_probs <- NULL
  
  for(m in mfinal_grid) {
    tryCatch({
      set.seed(123)
      if(mod_name == "Bagging") {
        fit <- bagging(rating_binary ~ ., data = df_train_scaled, mfinal = m, 
                       control = complex_control)
        pred <- predict.bagging(fit, newdata = df_test_scaled)
      } else {
        fit <- boosting(rating_binary ~ ., data = df_train_scaled, mfinal = m,
                        control = complex_control)
        pred <- predict.boosting(fit, newdata = df_test_scaled)
      }
      
      current_probs <- pred$prob[, 2]
      roc_obj <- suppressMessages(roc(df_test_scaled$rating_binary, current_probs))
      current_gini <- 2 * as.numeric(auc(roc_obj)) - 1
      
      print(paste("   > Trees:", m, "| Gini:", round(current_gini, 3)))
      
      if(current_gini > best_gini) {
        best_gini <- current_gini
        best_m <- m
        best_probs <- current_probs
      }
      rm(fit, pred); gc() # RAM Cleanup
      
    }, error = function(e) { print(paste("Error at m=", m)) })
  }
  
  # Calculate Metrics ONLY for the Winner
  if(!is.null(best_probs)) {
    metrics <- calc_metrics_tuned(df_test_scaled$rating_binary, best_probs)
    duration <- round(as.numeric(difftime(Sys.time(), start_time, units="secs")), 2)
    
    tuned_results[nrow(tuned_results)+1, ] <- c(
      mod_name, 
      paste("mfinal=", best_m, sep=""), 
      round(metrics["Accuracy"], 3), 
      round(metrics["TDL"], 3),
      round(metrics["Gini"], 3), 
      duration
    )
  }
}

# 4. FINAL OUTPUT
# ------------------------------------------------------------
tuned_results$Gini <- as.numeric(tuned_results$Gini)
tuned_results$TDL <- as.numeric(tuned_results$TDL)
tuned_results$Accuracy <- as.numeric(tuned_results$Accuracy)
tuned_results$Time_Sec <- as.numeric(tuned_results$Time_Sec)

# Create Master Table
  # Fix Base table types
  results_table$Gini <- as.numeric(results_table$Gini)
  results_table$TDL <- as.numeric(results_table$TDL)
  results_table$Accuracy <- as.numeric(results_table$Accuracy)
  results_table$Time_Sec <- as.numeric(results_table$Time_Sec)
  
  # Add labels
  results_table$Configuration <- "Base (Default)"
  tuned_results$Configuration <- paste("Tuned (", tuned_results$Best_Param, ")", sep="")
  
  # Bind
  master_table <- bind_rows(
    results_table %>% dplyr::select(Model, Configuration, Gini, TDL, Accuracy, Time_Sec),
    tuned_results %>% dplyr::select(Model, Configuration, Gini, TDL, Accuracy, Time_Sec)
  ) %>%
    arrange(desc(Gini))
  
  colnames(master_table) <- c("Algorithm", "Configuration", "Gini Coeff.", "Top Decile Lift", "Accuracy", "Training Time (s)")
  
  print("--- Final Table ---")
  print(master_table)
  
  # Export to CSV to copy into Word easily
  write.csv(master_table, "Table2_ML_Race.csv", row.names = FALSE)
  print("Table saved as 'Table2_ML_Race.csv'")
  
  
  