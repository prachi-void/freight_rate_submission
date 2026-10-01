# Freight Rate Prediction Challenge — Submission

## Files
- `train_and_predict.py` — feature engineering, CatBoost training, final predictions.
- `score.py` — supplied scorer logic; validates both required outputs and creates the December chart.
- `validation_predictions.csv` — 12,000 final predictions.
- `data/december_chart_inputs.csv` — 31 completed December predictions.
- `scorer_results/candidate_december.png` — fixed December prediction chart.
- `report.docx` — assessment report.
- `requirements.txt` — Python dependencies.

## Reproduce

```bash
python -m pip install -r requirements.txt
python train_and_predict.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The training script expects:
- `data/train_test.csv`
- `data/validation.csv`
- `data/december_chart_inputs.csv`

## Validation strategy

The labeled development data covers January–October 2025, while the final validation set covers November–December 2025. Therefore October is used as the primary forward-looking holdout. A random 80/20 split is also reported as a secondary sanity check.

The model is CatBoost regression trained on `log1p(posted_rate)`. Features include route/location, distance, weight, equipment, market/quote signals, calendar features and cyclical day-of-year/day-of-week features. Missing numeric values are median-imputed with missingness indicators.

For the December chart, the supplied chart file contains only pickup, delivery, distance, equipment, weight and date. Pickup/delivery coordinates are reconstructed from the labeled city data; unavailable market/quote signals are imputed from their training medians. This keeps the main model schema consistent while respecting the chart input contract.

## Submission note

The final hidden/test score is not available locally; `score.py` only validates file structure and creates the required chart.
