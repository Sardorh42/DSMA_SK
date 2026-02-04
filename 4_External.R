library(dplyr)
library(lubridate)

# ============================================================
# Weather Data
# ============================================================

# 1. Define the URL for Philadelphia Intl Airport (USW00013739)
# i request TMAX, TMIN and PRCP (Rain)
base_url <- "https://www.ncei.noaa.gov/access/services/data/v1"
query <- paste0(
  "?dataset=daily-summaries",
  "&stations=USW00013739",        # This is the code for Philly Airport
  "&startDate=2012-01-01",
  "&endDate=2022-12-31",
  "&dataTypes=TMAX,TMIN,PRCP",
  "&format=csv",
  "&includeStationName=true"
)

full_url <- paste0(base_url, query)

# 2. Download the data directly into R
print("Downloading Philadelphia Weather Data... (This takes a few seconds)")
weather_philly <- read.csv(full_url)

# 3. Clean it up (Fix date formats)
weather_philly$DATE <- as.Date(weather_philly$DATE)
# NOAA data is in Tenths of a degree (e.g., 200 = 20.0°C). We fix this now.
weather_philly$TMAX <- weather_philly$TMAX / 10
weather_philly$TMIN <- weather_philly$TMIN / 10

# ============================================================
# DATA INTEGRATION (MERGING)
# Concept: Enriching internal data with external attributes 
# ============================================================

# 1. Ensure Date Formats Match 
# The 'review_date' in df_final and 'DATE' in weather_philly must be identical types.

# Check current format of review_date
# If it was loaded from CSV, it might be a character string (e.g., "2018-05-20")
df_final$review_date <- as.Date(df_final$review_date)

# Check weather date (ensure it was cleaned in the previous step)
weather_philly$DATE <- as.Date(weather_philly$DATE)

# 2. Perform the Merge (Left Join)
# i join 'weather_philly' onto 'df_final' using the date columns.
print("Merging Yelp Reviews with Weather Data...")

df_merged <- df_final %>%
  left_join(weather_philly, by = c("review_date" = "DATE"))

# 3. Post-Merge Diagnostic 
# Check if the merge created any NAs (which would imply missing weather days)
missing_weather <- sum(is.na(df_merged$TMAX))

if(missing_weather > 0) {
  warning(paste("Alert:", missing_weather, "reviews are missing weather data."))
  # Imputation strategy for weather: 
  # If missing, it's often safe to assume 0 precipitation (PRCP) or fill Temp with previous day.
  # For now, i will inspect the percentage.
  print(paste("Percentage missing:", round(missing_weather/nrow(df_merged)*100, 2), "%"))
} else {
  print("Success! All reviews matched with weather data.")
}

# 4. Feature Engineering: "Bad Weather" Flag
# Instead of just raw numbers, i create a feature that might actually drive behavior.
df_merged <- df_merged %>%
  mutate(
    # Handle NA in PRCP (Precipitation) by assuming 0 (no rain) if missing
    PRCP = ifelse(is.na(PRCP), 0, PRCP),
    # Create binary flags (easier for Decision Trees/Random Forest later)
    Is_Rainy = ifelse(PRCP > 0, 1, 0),
    Is_Freezing = ifelse(TMIN < 0, 1, 0) # TMIN is in Celsius
  )

# 5. Save the Master Analysis File
write.csv(df_merged, "Philadelphia_Final_Analysis_Set.csv", row.names = FALSE)

print("Merged dataset saved as 'Philadelphia_Final_Analysis_Set.csv'")