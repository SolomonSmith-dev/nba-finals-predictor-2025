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

import matplotlib.pyplot as plt

# Bar chart of championship probabilities
teams = df_predictions["Team"]
probs = df_predictions["Championship_Probability"]

plt.figure(figsize=(10, 6))
plt.barh(teams, probs)
plt.xlabel("Championship Probability")
plt.title("2025 NBA Finals Predictions")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("data/processed/final_four_plot.png")
plt.show()

# 🆕 Add below this line:
from src.model import simulate_tournament

print("\n🎲 Running 10,000 tournament simulations...")
simulated_results = simulate_tournament(df_predictions)
print(simulated_results)

# Plot simulated outcomes
simulated_results.sort_values().plot(kind="barh", figsize=(10, 6), title="Simulated Champion Odds (10,000 runs)")
plt.xlabel("Win %")
plt.tight_layout()
plt.savefig("data/processed/simulated_champion_distribution.png")
plt.show()

from src.analysis import analyze_team_performance

# 🔧 Temporary mock playoff performance data
playoff_stats_df = pd.DataFrame([
    {
        "Team": "Oklahoma City Thunder",
        "OffRtg": 118.2,
        "DefRtg": 107.5,
        "REB%": 51.3,
        "TOV%": 12.1,
        "3P%": 36.7
    },
    {
        "Team": "New York Knicks",
        "OffRtg": 114.4,
        "DefRtg": 109.1,
        "REB%": 50.8,
        "TOV%": 13.8,
        "3P%": 34.6
    },
    {
        "Team": "Minnesota Timberwolves",
        "OffRtg": 113.0,
        "DefRtg": 105.4,
        "REB%": 49.5,
        "TOV%": 14.3,
        "3P%": 36.0
    },
    {
        "Team": "Indiana Pacers",
        "OffRtg": 117.2,
        "DefRtg": 113.9,
        "REB%": 48.2,
        "TOV%": 13.5,
        "3P%": 35.1
    }
])

# 🔍 Analyze each team’s playoff performance
for team in df_predictions["Team"]:
    analysis = analyze_team_performance(team, playoff_stats_df)
    print(f"\n📊 {team} Playoff Performance Analysis")
    print(f"✅ Strengths: {', '.join(analysis['Strengths']) if analysis['Strengths'] else 'None'}")
    print(f"⚠️ Weaknesses: {', '.join(analysis['Weaknesses']) if analysis['Weaknesses'] else 'None'}")
