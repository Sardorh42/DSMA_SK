# ============================================================
# STEP 6: RUNNING THE 9 SUPERVISED MODELS
# ============================================================

library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(nnet)
library(pROC)
# Mac fix
Sys.setenv(RGL_USE_NULL = TRUE)
library(adabag)

# 1. Define Helper Function for Metrics
# ------------------------------------------------------------
# Updated to handle "Negative"/"Positive" text labels
calc_metrics <- function(actual, predicted_prob) {
  
  # Convert actual "Negative"/"Positive" to 0/1 for calculation
  # Assuming "Positive" is the target (1)
  actual_numeric <- ifelse(actual == "Positive", 1, 0)
  
  dat <- data.frame(actual = actual_numeric, prob = predicted_prob)
  dat <- dat[order(-dat$prob), ] 
  
  # A. Accuracy (Threshold 0.5)
  pred_class_num <- ifelse(dat$prob > 0.5, 1, 0)
  acc <- mean(pred_class_num == dat$actual)
  
  # B. Top Decile Lift (TDL)
  top_10 <- floor(0.1 * nrow(dat))
  if(top_10 < 1) top_10 <- 1
  base_rate <- mean(dat$actual)
  top_rate  <- mean(dat$actual[1:top_10])
  
  tdl <- if(base_rate > 0) top_rate / base_rate else 0
  
  # C. Gini Coefficient
  roc_obj <- suppressMessages(roc(dat$actual, dat$prob))
  gini <- 2 * as.numeric(auc(roc_obj)) - 1
  
  return(c(Accuracy = acc, TDL = tdl, Gini = gini))
}

# 2. Setup Loop
# ------------------------------------------------------------
model_list <- c("Logit", "Naive Bayes", "KNN", "SVM", "Decision Tree", 
                "Bagging", "Boosting", "Random Forest", "Neural Net")

results_table <- data.frame(Model = model_list, Time_Sec = NA, Accuracy = NA, TDL = NA, Gini = NA)
model_formula <- as.formula("rating_binary ~ .")

print("Starting Model Loop with 'Positive' class target...")

# 3. The Loop
# ------------------------------------------------------------
for(i in 1:length(model_list)) {
  mod_name <- model_list[i]
  print(paste("Running:", mod_name, "..."))
  start_time <- Sys.time()
  probs <- NULL 
  
  tryCatch({
    if (mod_name == "Logit") {
      # Logit predicts probability of the second factor level ("Positive") by default
      fit <- glm(model_formula, data = df_train_scaled, family = "binomial")
      probs <- predict(fit, newdata = df_test_scaled, type = "response")
      
    } else if (mod_name == "Naive Bayes") {
      fit <- naiveBayes(model_formula, data = df_train_scaled)
      pred <- predict(fit, newdata = df_test_scaled, type = "raw")
      # FIX: Look for "Positive" instead of "1" (looking for "1" created some issues, hence the #FIX)
      probs <- pred[, "Positive"]
      
    } else if (mod_name == "KNN") {
      fit <- knn3(model_formula, data = df_train_scaled, k = 10)
      pred <- predict(fit, newdata = df_test_scaled, type = "prob")
      # FIX: Look for "Positive"
      probs <- pred[, "Positive"]
      
    } else if (mod_name == "SVM") {
      fit <- svm(model_formula, data = df_train_scaled, probability = TRUE)
      pred <- predict(fit, newdata = df_test_scaled, probability = TRUE)
      # FIX: Look for "Positive"
      probs <- attr(pred, "probabilities")[, "Positive"]
      
    } else if (mod_name == "Decision Tree") {
      fit <- rpart(model_formula, data = df_train_scaled, method = "class")
      pred <- predict(fit, newdata = df_test_scaled, type = "prob")
      # FIX: Look for "Positive"
      probs <- pred[, "Positive"]
      
    } else if (mod_name == "Bagging") {
      fit <- bagging(model_formula, data = df_train_scaled, mfinal = 5)
      pred <- predict.bagging(fit, newdata = df_test_scaled)
      # adabag returns probs for all classes. 
      # Usually column 2 is the second level (Positive)
      probs <- pred$prob[, 2]
      
    } else if (mod_name == "Boosting") {
      fit <- boosting(model_formula, data = df_train_scaled, mfinal = 5)
      pred <- predict.boosting(fit, newdata = df_test_scaled)
      probs <- pred$prob[, 2]
      
    } else if (mod_name == "Random Forest") {
      fit <- randomForest(model_formula, data = df_train_scaled, ntree = 300)
      pred <- predict(fit, newdata = df_test_scaled, type = "prob")
      # FIX: Look for "Positive"
      probs <- pred[, "Positive"]
      
    } else if (mod_name == "Neural Net") {
      # nnet with size=1 and decay (basic setup)
      # Capture output to avoid clutter
      capture.output(fit <- nnet(model_formula, data = df_train_scaled, size = 3, trace = FALSE))
      probs <- predict(fit, newdata = df_test_scaled, type = "raw")
    }
    
    end_time <- Sys.time()
    
    # Calculate Metrics using the Fixed Function
    metrics <- calc_metrics(df_test_scaled$rating_binary, probs)
    
    results_table[i, "Time_Sec"] <- round(as.numeric(difftime(end_time, start_time, units = "secs")), 2)
    results_table[i, 3:5]        <- round(metrics, 3)
    
  }, error = function(e) {
    print(paste("Error in", mod_name, ":", e$message))
  })
}

# 4. View Final Results
print("--- Final Model Comparison ---")
print(results_table)

# Identify Champion
best_model <- results_table[which.max(results_table$Gini), ]
print(paste("Champion Model:", best_model$Model, "with Gini =", best_model$Gini))