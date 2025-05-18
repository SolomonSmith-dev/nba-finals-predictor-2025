# NBA Finals Predictor 2025

Predicts the 2025 NBA Finals champion based on current season and playoff data using machine learning.

## Structure

- `src/`: Source code (scraping, preprocessing, modeling)
- `data/`: Raw and processed data
- `main.py`: Run full pipeline
- `notebooks/`: Exploration notebooks

## Usage

```bash
python main.py


---

### 2. `requirements.txt`
To auto-generate or update:
```bash
pip freeze > requirements.txt


## 🧪 Test Checklist

- [ ] Can run `main.py` end-to-end with no errors
- [ ] `data/raw/` contains scraped files
- [ ] `data/processed/` contains:
  - Cleaned team features
  - Model predictions
  - Visual bar chart of probabilities
  - Simulated 10,000-run tournament output
- [ ] Console shows sorted probabilities
- [ ] Plots display and save to disk
