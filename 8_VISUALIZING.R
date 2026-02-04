library(ggplot2)
library(ggrepel)
library(dplyr)
library(tidyr)
# ============================================================
# STEP 8.1: VISUALIZING TUNING IMPACT
# ============================================================

# 1. Prepare Data
plot_data_clean <- master_table %>%
  mutate(
    Algorithm_Clean = case_when(
      grepl("Logit", Algorithm) ~ "Logit",
      grepl("Tree", Algorithm) | grepl("rpart", Algorithm) ~ "Tree",
      grepl("SVM", Algorithm, ignore.case = TRUE) ~ "SVM",
      grepl("Neural", Algorithm) | grepl("nnet", Algorithm) ~ "Neural Net",
      grepl("Forest", Algorithm) | grepl("rf", Algorithm) ~ "Random Forest",
      TRUE ~ Algorithm
    ),
    Type = ifelse(grepl("Base", Configuration), "Base", "Tuned"),
    Gini = as.numeric(`Gini Coeff.`)
  ) %>%
  dplyr::select(Algorithm_Clean, Type, Gini) %>%
  pivot_wider(names_from = Type, values_from = Gini) %>%
  filter(!is.na(Base) & !is.na(Tuned)) %>%
  mutate(
    # Create a Status variable for coloring
    Status = ifelse(Tuned >= Base, "Improved", "Worsened (Overfit)"),
    Lift = Tuned - Base
  )

# 2. Generate Plot
ggplot(plot_data_clean, aes(x = Base, y = Tuned)) +
  
  # A. The Reference Line (Grey, subtle)
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey60") +
  
  # B. The Points (Colored by Success/Failure)
  # alpha=0.8 makes them slightly transparent so grid lines show through
  geom_point(aes(color = Status, size = abs(Lift)), alpha = 0.9) +
  
  # C. The Labels 
  geom_text_repel(aes(label = Algorithm_Clean), 
                  size = 4, 
                  color = "black", # Keep text black for contrast
                  box.padding = 0.5,
                  point.padding = 0.3,
                  min.segment.length = 0) +
  
  # D. Functional Color Palette 
  scale_color_manual(values = c("Improved" = "#009E73",        #Teal/Green
                                "Worsened (Overfit)" = "#ff6666")) + #Red/Orange
  
  # E. Formatting
  coord_fixed(ratio = 1) + 
  scale_size(range = c(3, 8), guide = "none") + # Hide size legend, keep color legend
  xlim(0.5, 0.85) + 
  ylim(0.5, 0.85) +
  theme_minimal() +
  labs(
    title = "Hyperparameter Tuning Impact",
    subtitle = "Models above the dashed line improved. Models below exhibited overfitting.",
    x = "Base Gini (Default)",
    y = "Tuned Gini (Optimized)",
    color = "Result"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    axis.title = element_text(face = "bold")
  )

# ============================================================
# STEP 8.2: MULTI-METRIC COMPARISON (TUNED MODELS ONLY)
# ============================================================


# 1. Filter and Prepare Data
# i select only the "Tuned" rows and the three key metrics
tuned_comparison <- master_table %>%
  filter(grepl("Tuned", Configuration)) %>% # Filter for Tuned models
  mutate(
    # Clean Algorithm names for the chart
    Algorithm_Clean = case_when(
      grepl("Logit", Algorithm) ~ "Logit",
      grepl("Tree", Algorithm) | grepl("rpart", Algorithm) ~ "Decision Tree",
      grepl("SVM", Algorithm, ignore.case = TRUE) ~ "SVM",
      grepl("Neural", Algorithm) | grepl("nnet", Algorithm) ~ "Neural Net",
      grepl("Forest", Algorithm) | grepl("rf", Algorithm) ~ "Random Forest",
      TRUE ~ Algorithm
    ),
    # Ensure all metrics are numeric
    TDL = as.numeric(`Top Decile Lift`),
    Gini = as.numeric(`Gini Coeff.`),
    Accuracy = as.numeric(Accuracy)
  ) %>%
  dplyr::select(Algorithm_Clean, TDL, Gini, Accuracy)

# 2. i create a list of models ordered specifically by their Gini score
gini_rank <- tuned_comparison %>%
  arrange(Gini) %>% # Arrange ascending (lowest first)
  pull(Algorithm_Clean)

# 3. Reshape to "Long" format for Faceted Plotting
long_metrics <- tuned_comparison %>%
  pivot_longer(cols = c(TDL, Gini, Accuracy), 
               names_to = "Metric", 
               values_to = "Value")

# 4. APPLY THE SORT ORDER
# i force the 'Algorithm_Clean' column to be a Factor with the levels i defined above
# This overrides the numeric values of TDL
long_metrics$Algorithm_Clean <- factor(long_metrics$Algorithm_Clean, levels = gini_rank)

# 5. Define Colors
prof_colors <- c("Gini" = "#009E73",      # Teal (Primary Sort)
                 "TDL" = "#D55E00",       # Orange
                 "Accuracy" = "#0072B2")  # Blue

# 6. Generate the Faceted Chart
ggplot(long_metrics, aes(x = Algorithm_Clean, y = Value, fill = Metric)) + # Removed reorder() here
  
  geom_bar(stat = "identity", width = 0.7, show.legend = FALSE) +
  
  # Labels
  geom_text(aes(label = sprintf("%.3f", Value)), 
            hjust = -0.1, size = 3.5, fontface = "bold", color = "grey30") +
  
  # Faceting
  facet_wrap(~Metric, scales = "free_x") + 
  
  coord_flip() +
  
  scale_fill_manual(values = prof_colors) +
  
  theme_bw() +
  labs(
    title = "Tuned Model Performance",
    subtitle = "Note: Sorted by Gini Coefficient (Center Panel). Check TDL for profitability.",
    x = "",
    y = "Score"
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold", size = 12, color = "white"),
    strip.background = element_rect(fill = "grey30"),
    axis.text.y = element_text(size = 10, face = "bold"),
    panel.grid.major.y = element_blank()
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2)))