import pandas as pd

# These are the 4 teams left in the playoffs
FINAL_FOUR = ["Indiana Pacers", "New York Knicks", "Oklahoma City Thunder", "Minnesota Timberwolves"]

# Mapping to match Basketball-Reference abbreviations if needed
TEAM_NAME_FIXES = {
    "Indiana": "Indiana Pacers",
    "New York": "New York Knicks",
    "Oklahoma City": "Oklahoma City Thunder",
    "Minnesota": "Minnesota Timberwolves"
}

def load_and_filter_team_stats(stats_path: str):
    """Load team stats, filter for final 4 teams, and clean up."""
    df = pd.read_csv(stats_path)

    # Clean team names
    df["Team"] = df["Team"].str.strip()
    df["Team"] = df["Team"].replace(TEAM_NAME_FIXES)

    # Filter only for final 4
    filtered = df[df["Team"].isin(FINAL_FOUR)].copy()

    # Select useful features
    columns_to_keep = ["Team", "W", "L", "W/L%", "ORtg", "DRtg", "NetRtg", "Pace"]
    filtered = filtered[columns_to_keep]
    return filtered

def add_manual_seed_info(df: pd.DataFrame):
    """Manually add seed info (we could scrape this later too)."""
    seed_map = {
        "Indiana Pacers": 6,
        "New York Knicks": 2,
        "Oklahoma City Thunder": 1,
        "Minnesota Timberwolves": 3
    }
    df["Seed"] = df["Team"].map(seed_map)
    return df

def preprocess_team_data(stats_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline for 2025 team data."""
    df = load_and_filter_team_stats(stats_path)
    df = add_manual_seed_info(df)
    return df
