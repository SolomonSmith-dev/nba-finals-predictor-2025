import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample training data (mock logic)
def load_mock_training_data() -> pd.DataFrame:
    data = [
        # WinPct, PointsPerGame, FieldGoalPct, Seed, Champion
        [0.73, 118.0, 0.49, 1, 1],  # strong champion
        [0.67, 116.5, 0.48, 2, 1],
        [0.65, 114.0, 0.47, 3, 0],
        [0.61, 111.0, 0.46, 2, 0],
        [0.60, 112.5, 0.45, 4, 0],
        [0.55, 110.0, 0.44, 5, 0],
        [0.53, 108.2, 0.43, 6, 0],
        [0.51, 106.5, 0.43, 8, 0],
        [0.70, 117.8, 0.48, 1, 1],
        [0.68, 116.7, 0.48, 1, 1],
    ]
    columns = ["WinPct", "PointsPerGame", "FieldGoalPct", "Seed", "Champion"]
    return pd.DataFrame(data, columns=columns)

# Train the model
def train_champion_model(df: pd.DataFrame):
    X = df[["WinPct", "PointsPerGame", "FieldGoalPct", "Seed"]]
    y = df["Champion"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler

# Predict final 4 probabilities
def predict_final_four_chances(model, scaler, final_four_df: pd.DataFrame) -> pd.DataFrame:
    X_final = final_four_df[["WinPct", "PointsPerGame", "FieldGoalPct", "Seed"]]
    X_scaled = scaler.transform(X_final)

    probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being Champion

    results = final_four_df.copy()
    results["Championship_Probability"] = probs

    return results.sort_values(by="Championship_Probability", ascending=False)

# Simulate tournament 10,000 times
def simulate_tournament(df_predictions: pd.DataFrame, n_simulations: int = 10000) -> pd.Series:
    """Run Monte Carlo simulations based on predicted championship probabilities."""
    teams = df_predictions["Team"].tolist()
    probs = np.array(df_predictions["Championship_Probability"].values)

    winners = np.random.choice(teams, size=n_simulations, p=probs / probs.sum())
    return pd.Series(winners).value_counts(normalize=True)
