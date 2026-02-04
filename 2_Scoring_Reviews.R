library(dplyr)
library(readr)
library(reticulate)

# ============================================================
# PART A: Python (VADER Sentiment & Clean Word Count)
# ============================================================

py_run_string("
import pandas as pd
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

# Setup VADER
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()
tqdm.pandas()

# Access Data from R
reviews = r.df_master

# 1. Improved Word Count
def get_clean_word_count(text):
    # Handle non-string (Missing/NA) data explicitly [Lecture 5]
    if not isinstance(text, str): 
        return 0
    
    # Remove non-alphanumeric but keep numbers (e.g., '5 stars')
    clean = re.sub(r'[^a-zA-Z0-9\\s]', '', text.lower())
    return len(clean.split())

print('Counting words on CLEAN text...')
reviews['word_count'] = reviews['review_text'].progress_apply(get_clean_word_count)

# 2. Safer VADER Scoring
def get_vader_score(text):
    # If missing, return 0 (Neutral) or None (to filter later)
    if not isinstance(text, str):
        return 0.0
    return sid.polarity_scores(text)['compound']

print('Scoring Sentiment on RAW text...')
# Apply function to handle NAs correctly
reviews['sentiment_score'] = reviews['review_text'].progress_apply(get_vader_score)
")

# Update R dataframe
df_master <- py$reviews

# ============================================================
# PART B: Create Predictive Target
# ============================================================

# 1. Visualize Distribution (Check for J-Shape)
hist(df_master$review_rating, main="Distribution of Ratings", col="blue")

# 2. Define Binary Target
# STRATEGY: Use Median/Mean to ensure better Class Balance (Note:as both would lead to 4&5 stars being positive this is for information purposes only)

# Remove rows where rating is NA before calculation
df_master <- df_master[!is.na(df_master$review_rating), ]

# Option A: Mean Split 
global_mean <- mean(df_master$review_rating)
print(paste("Global Mean:", round(global_mean, 2)))

# Option B: Median Split 
global_median <- median(df_master$review_rating)
print(paste("Global Median:", global_median))

# Create Target (check balance!)
df_master$rating_binary <- ifelse(df_master$review_rating > global_mean, 1, 0)

# 3. CRITICAL: Check Class Balance
print("Class Balance:")
balance <- prop.table(table(df_master$rating_binary))
print(balance)

# Warning if imbalance is severe (> 70/30 split)
if(max(balance) > 0.70) {
  warning("ALERT: Severe Class Imbalance detected. Consider using SMOTE or Under-sampling as per Lecture 5.")
}

# ============================================================
# Flatten List Columns (Data Cleaning)
# ============================================================

# Identify which columns are lists (likely 'categories' or 'attributes' if they weren't fully cleaned)
list_cols <- sapply(df_master, is.list)

# If any list columns exist, convert them to comma-separated strings
if(any(list_cols)) {
  print(paste("Flattening list columns:", names(df_master)[list_cols]))
  
  # Apply a function to convert c("A", "B") into "A, B"
  df_master[list_cols] <- lapply(df_master[list_cols], function(col) {
    sapply(col, function(cell) paste(unlist(cell), collapse = ", "))
  })
}

# ============================================================
# SAVE
# ============================================================

write.csv(df_master, "Philadelphia_With_Sentiment_And_Target.csv", row.names = FALSE)
print("Success! File saved.")