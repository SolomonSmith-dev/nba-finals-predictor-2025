import pandas as pd

def analyze_team_performance(team_name: str, playoff_stats: pd.DataFrame) -> dict:
    """
    Analyze playoff performance for a given team using playoff stat thresholds.

    Parameters:
        team_name (str): Name of the team.
        playoff_stats (pd.DataFrame): DataFrame containing playoff statistics for all teams.

    Returns:
        dict: Dictionary containing strengths and weaknesses.
    """
    team_stats = playoff_stats[playoff_stats['Team'] == team_name]

    if team_stats.empty:
        return {'Team': team_name, 'Strengths': [], 'Weaknesses': ['No data available']}

    row = team_stats.iloc[0]
    strengths = []
    weaknesses = []

    # Adjusted to your actual stat column names
    if row.get("OffRtg", 0) > 115:
        strengths.append("High Offensive Rating")
    else:
        weaknesses.append("Below-average Offense")

    if row.get("DefRtg", 999) < 110:
        strengths.append("Strong Defense")
    else:
        weaknesses.append("Weak Defense")

    if row.get("REB%", 0) > 50:
        strengths.append("Strong Rebounding")
    else:
        weaknesses.append("Rebounding Concerns")

    if row.get("TOV%", 100) < 13:
        strengths.append("Low Turnover Rate")
    else:
        weaknesses.append("High Turnover Rate")

    if row.get("3P%", 0) > 35:
        strengths.append("Good 3-Point Shooting")
    else:
        weaknesses.append("Poor 3-Point Shooting")

    return {
        'Team': team_name,
        'Strengths': strengths,
        'Weaknesses': weaknesses
    }
