import os
import pandas as pd
from src.scrape import scrape_2025_team_stats, scrape_2025_playoffs
from src.preprocess import preprocess_team_data
from src.model import (
    load_mock_training_data,
    train_champion_model,
    predict_final_four_chances,
)

# Ensure necessary folders exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Scrape and save raw data
df_stats = scrape_2025_team_stats()
df_playoffs = scrape_2025_playoffs()

df_stats.to_csv("data/raw/team_stats_2025.csv", index=False)
df_playoffs.to_csv("data/raw/playoffs_2025.csv", index=False)
print("✅ Scraping complete. Files saved in data/raw/")

# Preprocess the scraped team stats
df_final_four = preprocess_team_data("data/raw/team_stats_2025.csv")
df_final_four.to_csv("data/processed/final_four_features.csv", index=False)
print("✅ Preprocessing complete. Final 4 team features saved in data/processed/")

# Train the model on mock championship data
mock_training_data = load_mock_training_data()
model, scaler = train_champion_model(mock_training_data)

# Predict championship probabilities for the final four teams
df_predictions = predict_final_four_chances(model, scaler, df_final_four)
df_predictions.to_csv("data/processed/final_four_predictions.csv", index=False)

# Print final results
print("\n🏀 2025 NBA Finals Prediction Results:")
print(df_predictions[["Team", "Championship_Probability"]])
