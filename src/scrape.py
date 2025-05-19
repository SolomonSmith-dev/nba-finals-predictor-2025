import requests
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag

def scrape_2025_team_stats() -> pd.DataFrame:
    """Scrape 2025 regular season team per-game stats from Basketball Reference."""
    url = "https://www.basketball-reference.com/leagues/NBA_2025.html"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch team stats. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "lxml")

    # Use the per-game team stats table (more reliable)
    per_game_table = soup.find("table", {"id": "per_game-team"})
    if not per_game_table:
        raise Exception("Could not find per-game team stats table on the page.")

    df = pd.read_html(str(per_game_table))[0]
    df = df[df["Team"] != "League Average"]
    df["Team"] = df["Team"].str.replace(r"\*", "", regex=True)

    return df

def scrape_2025_playoffs() -> pd.DataFrame:
    """Scrape current 2025 NBA playoff matchups and results."""
    url = "https://www.basketball-reference.com/playoffs/NBA_2025.html"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch playoff data. Status code: {response.status_code}")
    
    soup = BeautifulSoup(response.content, "lxml")

    series_blocks = soup.find_all("div", attrs={"class": "series_summary"})
    series_results: list[dict[str, str]] = []

    for block in series_blocks:
        if not isinstance(block, Tag):
            continue  # Skip if not a Tag object

        teams = block.find_all("a")
        text = block.get_text(separator=" ").strip()

        if len(teams) == 2:
            team1 = teams[0].text
            team2 = teams[1].text
            series_results.append({
                "Matchup": f"{team1} vs {team2}",
                "Result": text
            })

    return pd.DataFrame(series_results)
