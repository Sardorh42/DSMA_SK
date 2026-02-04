library(dplyr)
library(tidyr)
library(stringr)
library(mice)
library(tm)
library(wordcloud)
library(RColorBrewer)
# 1. Load Data
df_scored <- read.csv("Philadelphia_With_Sentiment_And_Target.csv", stringsAsFactors = FALSE)

# ============================================================
# STEP 1: OUTLIER REMOVAL (Statistical Filter)
# Concept: Trimming based on IQR
# ============================================================

# Calculate Quartiles and IQR
Q1 <- quantile(df_scored$word_count, 0.25, na.rm = TRUE)
Q3 <- quantile(df_scored$word_count, 0.75, na.rm = TRUE)
IQR_value <- Q3 - Q1

# Define 1.5x IQR Threshold 
upper_limit <- Q3 + 1.5 * IQR_value

print(paste("Removing reviews longer than", round(upper_limit, 0), "words."))

# Apply Trimming
# i remove extreme outliers to avoid skewing the model
df_no_outliers <- df_scored %>%
  filter(word_count <= upper_limit & word_count > 0)

print(paste("Rows removed:", nrow(df_scored) - nrow(df_no_outliers)))

# ============================================================
# STEP 2: ATTRIBUTE STANDARDIZATION (Data Wrangling)
# Concept: Standardizing and Type Checking 
# ============================================================

# Function to clean string artifacts (u'string')
clean_attr_str <- function(x) {
  if (all(is.na(x))) return(x)
  # Remove u' prefix and surrounding quotes
  x <- str_remove(x, "^u'") 
  x <- str_remove_all(x, "'")
  # Convert "None" strings to true NA
  x[x == "None"] <- NA
  return(x)
}

df_clean <- df_no_outliers %>%
  mutate(
    # --- 1. Clean String Artifacts ---
    NoiseLevel = clean_attr_str(NoiseLevel),
    WiFi = clean_attr_str(WiFi),
    Alcohol = clean_attr_str(Alcohol),
    OutdoorSeating = clean_attr_str(OutdoorSeating),
    PriceRange = clean_attr_str(PriceRange),
    
    # --- 2. Map to Numeric/Logical (Standardization) ---
    # Lecture Warning: Do NOT replace NA with '2' (Mean Substitution)
    NoiseLevel_Num = case_when(
      NoiseLevel == "quiet" ~ 1,
      NoiseLevel == "average" ~ 2,
      NoiseLevel == "loud" ~ 3,
      NoiseLevel == "very_loud" ~ 4,
      TRUE ~ NA_real_ # Keep NA as NA to avoid bias
    ),
    
    # Binary variables
    Has_WiFi = ifelse(WiFi == "free", 1, 0),
    Has_Alcohol = ifelse(Alcohol != "none" & !is.na(Alcohol), 1, 0),
    Has_OutdoorSeating = ifelse(OutdoorSeating == "True", 1, 0),
    
    # Price Range: Ensure it is numeric
    # If missing, i keep as NA rather than guessing
    PriceRange_Num = as.numeric(PriceRange),
    
    # --- 3. Social Data ---
    # Fans is a count, so NAs can safely be 0 (absence of data = 0 fans)
    user_fans = replace_na(as.numeric(user_fans), 0)
  )

# ============================================================
# STEP 3: SANITY CHECK
# Concept: Verify data quality before analysis
# ============================================================

# Check for "Zero-Length" text issues
empty_count <- sum(nchar(df_clean$review_text) == 0, na.rm = TRUE)
print(paste("Empty reviews detected:", empty_count))

# Check proportion of NAs in key variables (Completeness)
print("Missing Value Analysis:")
print(colMeans(is.na(df_clean[, c("NoiseLevel_Num", "PriceRange_Num")])))

# 1. Handle PriceRange (3.5% missing) - Safe to drop
# i remove these rows because 3.5% < 5% threshold
df_final <- df_clean[!is.na(df_clean$PriceRange_Num), ]

# 2. Handle NoiseLevel (8% missing) - Impute 
# Instead of dropping, i will fill NAs with the median (integer) 
# MICE (Multiple Imputation by Chained Equations)
imputed_data <- mice(df_final, m=5, method='pmm', seed=500) # Predictive Mean Matching
#Check. Imputation process was successful
imputed_data$loggedEvents
df_final <- complete(imputed_data, 1)

print(paste("Rows remaining:", nrow(df_final)))

# 3. Final Check
print("Final Missing Value Check:")
print(colMeans(is.na(df_final[, c("NoiseLevel_Num", "PriceRange_Num")])))


# ============================================================
# VISUAL PROOF: Text as a Proxy for Missing Variables
# Context: Validating that 'sentiment_score' captures Service/Food
# Input: df_scored (from Source 81)
# ============================================================


# 1. Sample the data (Memory Management)
# i use a random sample of 5,000 reviews to ensure the plot generates quickly
set.seed(123)
if(nrow(df_scored) > 5000) {
  proxy_text <- sample(df_scored$review_text, 5000)
} else {
  proxy_text <- df_scored$review_text
}

# 2. Create Corpus
docs <- Corpus(VectorSource(proxy_text))

# 3. Standard Cleaning ONLY 
# i remove grammar/noise, but KEEP words like "food", "service", "waiter" (proxy for servce and food quality)
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("english")) # Only remove "the", "and", "is"

# 4. Generate the Matrix
dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m), decreasing=TRUE)
d <- data.frame(word = names(v), freq=v)

# 5. Generate the "Proxy" Word Cloud
set.seed(1234)
png("Proxy_WordCloud.png", width=800, height=800) # Save as image
wordcloud(words = d$word, 
          freq = d$freq, 
          min.freq = 50,
          max.words = 150, 
          random.order = FALSE, 
          rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"),
          scale = c(4, 0.5))
dev.off() # Close the image file

print("Word cloud generated. Check your working directory for 'Proxy_WordCloud.png'")