# Drivers of Satisfaction: Sentiment vs. Attributes in Philadelphia Restaurants

**Author:** Sardor Khusanov  
**Course:** Data Science and Marketing Analytics (Goethe University Frankfurt)  
**Date:** February 2026

## 📌 Project Overview
This repository contains the replication code for the seminar paper *"Sentiment Analysis"*. The study investigates whether unstructured semantic data (review sentiment) serves as a superior predictor of customer satisfaction compared to structured physical attributes (e.g., price, Wi-Fi) and environmental factors (weather).

**Methodology:** The project utilizes a hybrid analytical framework integrating SQL (DuckDB) for data extraction, Python (NLTK) for sentiment scoring, and R for machine learning and driver analysis.

## 📂 Repository Structure (Pipeline)

The analysis is executed sequentially through the following scripts:

### **Phase 1: Data Engineering**
*   **`1_Extract_Philly.R`**: Connects to the Yelp Open Dataset (JSON) via DuckDB to extract business and review data for Philadelphia restaurants (2012–2022).
*   **`2_Scoring_Reviews.R`**: Implements the **VADER** lexicon (via `reticulate`/Python) to calculate compound sentiment scores for raw review texts.
*   **`3_Cleaning.R`**: Performs data cleaning, including outlier detection (IQR method), median imputation for missing attributes.
*   **`4_External.R`**: Fetches historical weather data (precipitation, temperature) from the **NOAA API** and merges it with the review timestamps.

### **Phase 2: Preprocessing & Visualization**
*   **`5_Pre_Processing.R`**: Prepares the modeling dataset by applying **SMOTE** to handle class imbalance and scaling numerical features. 5.1 and 5.2 Generate exploratory visualizations, including the Word Cloud and Correlation Matrix.

### **Phase 3: Machine Learning & Evaluation**
*   **`6_ML.R`**: Trains baseline classifiers (All 9 supervised models).
*   **`7_Tuning.R`**: Executes Hyperparameter Optimization using Grid Search with 3-fold Cross-Validation.
*   **`8_VISUALIZING.R`**: Generates ML comparison tables as well as the Tuning Outcome Visualisation
*   **`9_Driver_Analysis.R`**: Performs the "Integrated Driver Analysis".
