import pandas as pd

FINAL_FOUR = ["Indiana Pacers", "New York Knicks", "Oklahoma City Thunder", "Minnesota Timberwolves"]

# Map shortened names to full names if necessary
TEAM_NAME_FIXES = {
    "Indiana": "Indiana Pacers",
    "New York": "New York Knicks",
    "Oklahoma City": "Oklahoma City Thunder",
    "Minnesota": "Minnesota Timberwolves"
}

def load_and_filter_team_stats(stats_path: str) -> pd.DataFrame:
    """Load per-game stats, filter for final 4 teams, and rename columns."""
    df = pd.read_csv(stats_path)

    # Clean team names
    df["Team"] = df["Team"].str.strip()
    df["Team"] = df["Team"].replace(TEAM_NAME_FIXES)

    # Filter for final 4
    filtered = df[df["Team"].isin(FINAL_FOUR)].copy()

    print("📊 Columns available:")
    print(filtered.columns.tolist())

    # Rename only columns we have
    filtered = filtered.rename(columns={
        "PTS": "PointsPerGame",
        "FG%": "FieldGoalPct"
    })

    # Add mock WinPct values manually
    win_pct_map = {
        "Indiana Pacers": 0.57,
        "New York Knicks": 0.61,
        "Oklahoma City Thunder": 0.69,
        "Minnesota Timberwolves": 0.68
    }
    filtered["WinPct"] = filtered["Team"].map(win_pct_map)

    # Select only columns we now have
    columns_to_keep = ["Team", "WinPct", "PointsPerGame", "FieldGoalPct"]
    filtered = filtered[columns_to_keep]

    return filtered


def add_manual_seed_info(df: pd.DataFrame) -> pd.DataFrame:
    """Manually add seed info for the 2025 playoffs."""
    seed_map = {
        "Indiana Pacers": 6,
        "New York Knicks": 2,
        "Oklahoma City Thunder": 1,
        "Minnesota Timberwolves": 3
    }
    df["Seed"] = df["Team"].map(seed_map)
    return df

def preprocess_team_data(stats_path: str) -> pd.DataFrame:
    df = load_and_filter_team_stats(stats_path)
    df = add_manual_seed_info(df)
    return df
