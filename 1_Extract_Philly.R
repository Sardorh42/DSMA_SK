#set working Directory. Please change
setwd("/Users/sardor/Desktop/DSMA")

library(duckdb)
library(DBI)

# 1. Define your city
CITY_NAME <- "Philadelphia"

# 2. Define the SQL Query
query <- sprintf("
SELECT
  -- REVIEW DATA
  r.review_id,
  r.stars AS review_rating,
  CAST(r.date AS DATE) AS review_date, -- Cast to Date type [Lecture 1]
  r.text AS review_text,
  r.useful AS useful_votes,
  r.user_id,

  -- BUSINESS DATA
  b.business_id,
  b.name AS restaurant_name,
  b.categories,

  -- SOCIAL DATA
  u.fans AS user_fans,
  -- Calculate count in SQL to avoid importing massive text strings
  CASE 
    WHEN u.friends = 'None' OR u.friends IS NULL THEN 0
    ELSE len(string_split(u.friends, ','))
  END AS max_friends_count,

  -- ATTRIBUTES
  json_extract_string(b.attributes, '$.NoiseLevel') AS NoiseLevel,
  json_extract_string(b.attributes, '$.WiFi') AS WiFi,
  json_extract_string(b.attributes, '$.Alcohol') AS Alcohol,
  json_extract_string(b.attributes, '$.OutdoorSeating') AS OutdoorSeating,
  json_extract_string(b.attributes, '$.BusinessAcceptsCreditCards') AS BusinessAcceptsCreditCards,
  json_extract_string(b.attributes, '$.RestaurantsDelivery') AS RestaurantsDelivery,
  json_extract_string(b.attributes, '$.RestaurantsPriceRange2') AS PriceRange,
  json_extract_string(b.attributes, '$.RestaurantsReservations') AS RestaurantsReservations,
  json_extract_string(b.attributes, '$.RestaurantsTableService') AS TableService

FROM read_json_auto('yelp_academic_dataset_review.json') r
JOIN read_json_auto('yelp_academic_dataset_business.json') b
  ON r.business_id = b.business_id
JOIN read_json_auto('yelp_academic_dataset_user.json') u
  ON r.user_id = u.user_id

WHERE b.city = '%s'
  AND b.categories LIKE '%%Restaurant%%'
  AND r.date >= '2012-01-01'
  AND r.date <= '2022-12-31'
", CITY_NAME)

# 3. Connect & Run
con <- dbConnect(duckdb::duckdb())
dbExecute(con, "INSTALL json; LOAD json;")

print("Running Master Extraction (Reviews + Social Counts + Attributes)...")
df_master <- dbGetQuery(con, query)

# 4. Process Derived Variables in R

# A. Covid Lockdown Binary
# Ensure the column is explicitly a Date object for comparison
df_master$review_date <- as.Date(df_master$review_date)

print("Generating Covid Lockdown Flag...")
df_master$covid_lockdown <- ifelse(
  df_master$review_date >= as.Date("2020-03-15") & df_master$review_date <= as.Date("2021-06-11"),
  1,
  0
)


# 5. Clean the Attributes
# handle NAs (Missing Data)

clean_attr <- function(x) {
  # If value is NA (missing), return NA or a placeholder like "Unknown"
  if (is.na(x)) return(NA) 
  
  x <- gsub("^u'", "", x) # Remove starting u'
  x <- gsub("'$", "", x)  # Remove ending '
  x <- gsub("'", "", x)   # Remove other quotes
  return(x)
}

# Apply cleaning to all attribute columns
attr_cols <- c("NoiseLevel", "WiFi", "Alcohol", "OutdoorSeating", "PriceRange", 
               "RestaurantsDelivery", "RestaurantsReservations", "TableService")

# Vectorized apply is safer than a loop for NAs in R
print("Cleaning Attributes...")
for(col in attr_cols) {
  if(col %in% colnames(df_master)) {
    # Use sapply to handle the NA check in the function correctly per element
    df_master[[col]] <- sapply(df_master[[col]], clean_attr)
  }
}

# 6. Save
output_filename <- paste0(CITY_NAME, "_reviews_full.csv")
write.csv(df_master, output_filename, row.names = FALSE)

print(paste("Success! Saved full dataset with", nrow(df_master), "rows to", output_filename))

dbDisconnect(con)
