library(tidyr)
library(ggplot2)
library(reshape2) 
library(dplyr)
library(lubridate)
library(caret)
library(tm)           
library(wordcloud)    
library(RColorBrewer) 
# ============================================================
# STEP 5: FEATURE ENGINEERING & SELECTION
# ============================================================

# 1. Load the Merged Data (from Step 4)
if(!exists("df_merged")) {
  df_merged <- read.csv("Philadelphia_Final_Analysis_Set.csv", stringsAsFactors = FALSE)
}

# 2. FEATURE ENGINEERING
# ------------------------------------------------------------

print("Engineering Temporal and Weather Variables...")

df_enhanced <- df_merged %>%
  mutate(
    # A. Date-Based Dummies 
    # -------------------------------------------------------
    review_date = as.Date(review_date),
    
    # "Weekend": 1 if Sat(7) or Sun(1), else 0
    day_num = wday(review_date), 
    is_weekend = ifelse(day_num %in% c(1, 7), 1, 0),
    
    # "Season": Derived from Month
    month_num = month(review_date),
    season = case_when(
      month_num %in% c(12, 1, 2) ~ "Winter",
      month_num %in% c(3, 4, 5) ~ "Spring",
      month_num %in% c(6, 7, 8) ~ "Summer",
      TRUE ~ "Fall"
    ),
    
    # "Covid": Already exists, but ensuring it's a factor
    covid_lockdown = as.factor(covid_lockdown),
    
    # B. Weather Variables
    # -------------------------------------------------------
    # TAVG: Average Temperature (Calculated from TMAX/TMIN)
    TAVG = (TMAX + TMIN) / 2,
    
    # PRCP: Ensure no NAs
    PRCP = ifelse(is.na(PRCP), 0, PRCP),
    
    # C. Service & Attribute Cleaning
    # -------------------------------------------------------
    # Function to clean "True"/"False" strings from JSON
    
    # Alcohol: Map to binary (1 = serves alcohol, 0 = no)
    Has_Alcohol = ifelse(!is.na(Alcohol) & Alcohol != "none" & Alcohol != "u'none'", 1, 0),
    
    # WiFi: Map to binary (1 = free/paid, 0 = no)
    Has_WiFi = ifelse(!is.na(WiFi) & grepl("free|paid", WiFi, ignore.case = TRUE), 1, 0),
    
    # Outdoor Seating
    Has_Outdoor = ifelse(!is.na(OutdoorSeating) & grepl("True", OutdoorSeating, ignore.case = TRUE), 1, 0),
    
    # Delivery
    Has_Delivery = ifelse(!is.na(RestaurantsDelivery) & grepl("True", RestaurantsDelivery, ignore.case = TRUE), 1, 0),
    
    # Reservations
    Has_Reservations = ifelse(!is.na(RestaurantsReservations) & grepl("True", RestaurantsReservations, ignore.case = TRUE), 1, 0),
    
    # Table Service
    Has_TableService = ifelse(!is.na(TableService) & grepl("True", TableService, ignore.case = TRUE), 1, 0),
    
    # Price Range: Ensure Numeric
    PriceRange_Num = as.numeric(PriceRange)
  )

# 3. HANDLING NOISE LEVEL (Ordinal Encoding)
# ------------------------------------------------------------
# Mapping text to numbers: quiet=1, average=2, loud=3, very_loud=4
df_enhanced$NoiseLevel_Num <- case_when(
  grepl("quiet", df_enhanced$NoiseLevel) ~ 1,
  grepl("average", df_enhanced$NoiseLevel) ~ 2,
  grepl("very_loud", df_enhanced$NoiseLevel) ~ 4, # Moved UP due to case_when issues
  grepl("loud", df_enhanced$NoiseLevel) ~ 3,      # Moved DOWN
  TRUE ~ NA_real_
)
# Impute missing NoiseLevel with Median
noise_med <- median(df_enhanced$NoiseLevel_Num, na.rm=TRUE)
df_enhanced$NoiseLevel_Num[is.na(df_enhanced$NoiseLevel_Num)] <- noise_med

# 4. FINAL SELECTION & CLEANUP
# ------------------------------------------------------------

# Define the exact list of variables 
keep_cols <- c(
  # Target
  "rating_binary",
  
  # Text Metrics
  "sentiment_score", "word_count", 
  
  # Social/User Metrics
  "user_fans", #"max_friends_count", removed for the ml model due to high correlation of theses two
  
  # Attributes (Internal)
  "PriceRange_Num", "NoiseLevel_Num", "Has_Alcohol", 
  "Has_Outdoor", "Has_WiFi", "Has_Delivery", 
  "Has_Reservations", "Has_TableService",
  
  # Context (External/Time)
  "TAVG", "PRCP", # Weather
  "covid_lockdown", "is_weekend", "season", "Is_Rainy", "Is_Freezing" # Dummies
)

# Select columns
df_model <- df_enhanced[, keep_cols]

# Drop rows with missing values (Listwise deletion for remaining NAs)
df_model <- na.omit(df_model)

# 5. FORMATTING FOR ML (Factor Conversion)
# ------------------------------------------------------------
# Machine Learning models in R (caret) require categorical vars as Factors

# Target Variable (Rename 0/1 to Negative/Positive for caret)
df_model$rating_binary <- factor(df_model$rating_binary, 
                                 levels = c(0, 1), 
                                 labels = c("Negative", "Positive"))

# Categorical Predictors
df_model$season <- as.factor(df_model$season)
df_model$covid_lockdown <- as.factor(df_model$covid_lockdown)
df_model$is_weekend <- as.factor(df_model$is_weekend)
df_model$Has_Alcohol <- as.factor(df_model$Has_Alcohol)
df_model$Has_Outdoor <- as.factor(df_model$Has_Outdoor)
df_model$Has_WiFi <- as.factor(df_model$Has_WiFi)
df_model$Has_Delivery <- as.factor(df_model$Has_Delivery)
df_model$Has_Reservations <- as.factor(df_model$Has_Reservations)
df_model$Has_TableService <- as.factor(df_model$Has_TableService)

# 6. DOWNSAMPLING (To speed up processing and RAM limitations)
# ------------------------------------------------------------
set.seed(123)
if(nrow(df_model) > 25000) {
  print("Downsampling to 25,000 rows for performance...")
  idx <- createDataPartition(df_model$rating_binary, p = 25000/nrow(df_model), list = FALSE)
  df_sample <- df_model[idx, ]
} else {
  df_sample <- df_model
}

# 7. TRAIN/TEST SPLIT & SCALING
# ------------------------------------------------------------
trainIndex <- createDataPartition(df_sample$rating_binary, p = 0.75, list = FALSE)
df_train <- df_sample[trainIndex, ]
df_test <- df_sample[-trainIndex, ]

# Scale Numeric Variables only (Normalization)
num_vars <- c("sentiment_score", "word_count", "user_fans", "PriceRange_Num", "TAVG", "PRCP")

preProc <- preProcess(df_train[, num_vars], method = c("center", "scale"))
df_train_scaled <- predict(preProc, df_train)
df_test_scaled <- predict(preProc, df_test)

print(colnames(df_train_scaled))

# ============================================================
# STEP 5.1: DESCRIPTIVE STATISTICS (Summary Table)
# Purpose: Generate "Data Overview" for the paper
# ============================================================

# ------------------------------------------------------------
# PART A: CONTINUOUS VARIABLES (Mean, Median, SD)
# ------------------------------------------------------------
# i select variables where "Average" makes sense.
# i include MEDIAN because 'user_fans' is highly skewed.

numeric_vars <- df_model %>%
  dplyr::select(
    sentiment_score,    # VADER Score
    word_count,         # Review Length
    user_fans,          # User Influence
    #max_friends_count,  # Social Network Size
    TAVG                # Temperature (Celsius)
  )

table_numeric <- numeric_vars %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable) %>%
  summarise(
    N = n(),
    Mean = round(mean(Value, na.rm = TRUE), 2),
    Median = round(median(Value, na.rm = TRUE), 2), # Key for skewed data
    Std_Dev = round(sd(Value, na.rm = TRUE), 2),
    Min = round(min(Value, na.rm = TRUE), 2),
    Max = round(max(Value, na.rm = TRUE), 2)
  ) %>%
  mutate(
    Variable_Label = case_when(
      Variable == "sentiment_score" ~ "Sentiment Score (-1 to +1)",
      Variable == "word_count" ~ "Review Length (Words)",
      Variable == "user_fans" ~ "User Fans (Count)",
      #Variable == "max_friends_count" ~ "Friend Count",
      Variable == "TAVG" ~ "Temperature (Celsius)",
      TRUE ~ Variable
    )
  ) %>%
  dplyr::select(Variable_Label, Mean, Median, Std_Dev, Min, Max)

print("--- TABLE 1A: CONTINUOUS VARIABLES ---")
print(table_numeric)
write.csv(table_numeric, "Table1a_Numeric_Stats.csv", row.names = FALSE)

# ------------------------------------------------------------
# PART B: CATEGORICAL / BINARY VARIABLES (Frequencies %)
# ------------------------------------------------------------
# i select variables where "Percentage" is the insight.
# This helps check Class Balance.

# 1. Select and label the data
cat_vars <- df_model %>%
  dplyr::select(
    rating_binary,      # Target
    PriceRange_Num,     # 1-4
    NoiseLevel_Num,     # 1-4
    is_weekend,         # 0/1
    Has_Alcohol,        # 0/1
    Has_Delivery,       # 0/1
    season,             # Winter/Spring/etc
    Is_Rainy,           # 0/1
    Is_Freezing         # 0/1
  )

# 2. Function to calculate % distribution
get_distribution <- function(data, col_name) {
  data %>%
    count(.data[[col_name]]) %>%
    mutate(
      Variable = col_name,
      Level = as.character(.data[[col_name]]),
      Percentage = round(n / sum(n) * 100, 1) # Convert to %
    ) %>%
    rename(Count = n) %>%
    dplyr::select(Variable, Level, Count, Percentage)
}

# 3. Loop through columns and bind rows
table_cat <- list()
for(col in colnames(cat_vars)) {
  table_cat[[col]] <- get_distribution(cat_vars, col)
}
final_cat_table <- bind_rows(table_cat)

# 4. Clean up Labels for the Report
final_cat_table$Variable <- case_when(
  final_cat_table$Variable == "rating_binary" ~ "Target: Positive Rating",
  final_cat_table$Variable == "PriceRange_Num" ~ "Price Tier (1=Cheap; 4=Exp)",
  final_cat_table$Variable == "NoiseLevel_Num" ~ "Noise Level (1=Quiet; 4=Very loud)",
  final_cat_table$Variable == "is_weekend" ~ "Weekend Visit",
  final_cat_table$Variable == "Has_Alcohol" ~ "Serves Alcohol",
  final_cat_table$Variable == "Has_Delivery" ~ "Offers Delivery",
  final_cat_table$Variable == "Is_Freezing" ~ "Freezing",
  final_cat_table$Variable == "Is_Rainy" ~ "Raining",
  TRUE ~ final_cat_table$Variable
)

print("--- TABLE 1B: CATEGORICAL DISTRIBUTIONS ---")
print(final_cat_table)
write.csv(final_cat_table, "Table1b_Categorical_Stats.csv", row.names = FALSE)

# ============================================================
# STEP 5.2: CORRELATION MATRIX
# Purpose: Check for Multicollinearity and Driver Strength
# ============================================================


print("--- Generating Correlation Matrix ---")

# 1. Select and Convert Variables
# I only choose numeric/ordinal variables where correlation "makes sense"
cor_data <- df_model %>%
  mutate(
    # Convert Factor Target back to Numeric (0 = Negative, 1 = Positive)
    Rating_Numeric = as.numeric(rating_binary) - 1
  ) %>%
  dplyr::select(
    Rating_Numeric,     # Target
    sentiment_score,    # The strongest predictor?
    word_count,         # Do longer reviews mean lower ratings?
    PriceRange_Num,     # Ordinal (1-4)
    NoiseLevel_Num,     # Ordinal (1-4)
    user_fans,          # Social influence
    #max_friends_count,  # Network size
    TAVG                # Temperature
  )

# 2. Calculate Correlation (Pearson)
# use = "complete.obs" ensures i don't crash on any remaining NAs
cor_matrix <- cor(cor_data, use = "complete.obs", method = "pearson")

# 3. Create a Clean Lower-Triangle Table. Just for info
cor_matrix_lower <- cor_matrix
cor_matrix_lower[upper.tri(cor_matrix_lower)] <- NA
cor_table <- round(cor_matrix_lower, 2) # Round to 2 decimals

print("---   CORRELATION MATRIX ---")
print(cor_table, na.print = "")

# 4. VISUALIZATION (Heatmap for the paper)
melted_cormat <- melt(cor_matrix)

ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) +
  geom_tile() +
  scale_fill_gradient2(low = "#D55E00", mid = "white", high = "#0072B2", 
                       midpoint = 0, limit = c(-1,1), name="Correlation") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 10, hjust = 1)) +
  coord_fixed() +
  geom_text(aes(Var2, Var1, label = round(value, 2)), color = "black", size = 3) +
  labs(
    title = "Correlation Heatmap of Key Variables",
    subtitle = "Blue = Positive Correlation, Red = Negative Correlation",
    x = "", y = ""  )

