import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sample training data (based on logic/intuition)
def load_mock_training_data() -> pd.DataFrame:
    data = [
        # Win%, NetRtg, Seed, Champion
        [0.73, 9.0, 1, 1],  # '16 Warriors
        [0.67, 6.5, 2, 1],  # '23 Nuggets
        [0.65, 5.8, 3, 0],
        [0.61, 3.5, 2, 0],
        [0.60, 2.8, 4, 0],
        [0.55, 1.1, 5, 0],
        [0.53, 0.2, 6, 0],
        [0.51, -1.0, 8, 0],
        [0.70, 7.2, 1, 1],  # '14 Spurs
        [0.68, 8.5, 1, 1],  # '13 Heat
    ]
    columns = ["WinPct", "NetRtg", "Seed", "Champion"]
    return pd.DataFrame(data, columns=columns)

# Train the model
def train_champion_model(df: pd.DataFrame):
    X = df[["WinPct", "NetRtg", "Seed"]]
    y = df["Champion"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler

# Predict final 4 probabilities
def predict_final_four_chances(model, scaler, final_four_df: pd.DataFrame) -> pd.DataFrame:
    X_final = final_four_df[["W/L%", "NetRtg", "Seed"]].copy()
    X_final.columns = ["WinPct", "NetRtg", "Seed"]

    X_scaled = scaler.transform(X_final)
    probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being Champion

    results = final_four_df.copy()
    results["Championship_Probability"] = probs

    return results.sort_values(by="Championship_Probability", ascending=False)
