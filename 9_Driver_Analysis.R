library(caret)
library(randomForest)
library(adabag)
library(dplyr)
library(ggplot2)

# ============================================================
# STEP 9: INTEGRATED DRIVER ANALYSIS
# Concept: Combining ML Importance with Logit Signs 
# ============================================================

print("--- Generating Driver Analysis Table ---")

# 1. GET MAGNITUDE (Random Forest Importance)
# -------------------------------------------
set.seed(123)
# I use a fast Random Forest just to get the variable importance
rf_driver_model <- train(rating_binary ~ ., 
                         data = df_train_scaled, 
                         method = "rf", 
                         importance = TRUE, 
                         trControl = trainControl(method = "none"), 
                         ntree = 100) 

# Extract Importance safely (Handling the "Overall" vs specific class issue)
imp_obj <- varImp(rf_driver_model, scale = TRUE)
rf_importance <- data.frame(
  Feature = rownames(imp_obj$importance),
  Importance = imp_obj$importance[, 1] # select the first column
)

# 2. GET DIRECTION (Logistic Regression Coefficients)
# -------------------------------------------
logit_driver_model <- glm(rating_binary ~ ., data = df_train_scaled, family = "binomial")

logit_coefs <- data.frame(
  Feature = names(coef(logit_driver_model)),
  Coefficient = coef(logit_driver_model)
)

# 3. MERGE AND FORMAT TABLE
# -------------------------------------------
driver_table <- merge(rf_importance, logit_coefs, by = "Feature") %>%
  # Create a descriptive "Impact" column based on the coefficient sign
  mutate(
    Impact_Direction = ifelse(Coefficient > 0, "Positive (Satisfier)", "Negative (Dissatisfier)"),
    # Round numbers for the paper
    Importance = round(Importance, 1),
    Coefficient = round(Coefficient, 3)
  ) %>%
  # Sort by Importance (Highest Magnitude first)
  arrange(desc(Importance)) %>%
  # Select and Rename columns for the paper
  dplyr::select(
    Feature, 
    Importance, 
    Coefficient, 
    Impact_Direction
  )

# 4. PRINT FINAL TABLE
# -------------------------------------------
print("--- TABLE 3: DETERMINANTS OF CUSTOMER SATISFACTION ---")
print(driver_table, row.names = FALSE)

# Export to CSV to copy into Word easily
write.csv(driver_table, "Table3_Driver_Analysis.csv", row.names = FALSE)
print("Table saved as 'Table3_Driver_Analysis.csv'")

# 5. VISUALIZATION (The Managerial Plot)
# ------------------------------------------------------------
# This visualizes the top 15 drivers identified by Random Forest
ggplot(head(driver_table, 15), aes(x = reorder(Feature, Importance), y = Importance, fill = Impact_Direction)) +
  geom_bar(stat = "identity", width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("Positive (Satisfier)" = "#009E73",  # green
                               "Negative (Dissatisfier)" = "#ff6666")) + # Red
  theme_minimal() +
  labs(
    title = "Determinants of Satisfaction (Random Forest Model)",
    subtitle = "Relative Importance of predictors (Bars) and their Impact Direction (Color)",
    x = "",
    y = "Relative Importance (0-100)",
    fill = "Impact Type"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.y = element_text(face = "bold", size = 10),
    legend.position = "bottom"
  )
