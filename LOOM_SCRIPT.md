# 2–3 Minute Loom Recording Script

## 0:00–0:20 — Problem
“This challenge asks us to predict freight posted rates for 12,000 November–December loads using 48,000 labeled January–October loads.”

## 0:20–0:50 — Validation
“Because the final prediction period is later in time, I used a chronological validation strategy. I trained on January through September and held out October. I also checked a random 80/20 split as a secondary sanity check, but the October forward split is the main validation.”

## 0:50–1:25 — Features and model
“I engineered route and city features, distance and geographic distance, weight, equipment, market and quote signals, calendar variables, and cyclical day-of-year and day-of-week features. Missing numeric values receive median imputation with explicit missingness flags. The model is CatBoost regression trained on log-transformed posted rates.”

## 1:25–1:50 — Results
“On the October holdout, the model achieved approximately 108.61 dollars MAE, 647.25 dollars RMSE, and R-squared of 0.8207. The random split was easier, with approximately 90.06 dollars MAE.”

## 1:50–2:20 — Final outputs
“After validation, I refit the model on all 48,000 labeled rows and predicted all 12,000 validation loads. I also filled the 31 December chart inputs. Since the chart file omits coordinates and two numeric signals, the pipeline reconstructs city coordinates from training data and uses development medians for unavailable signals.”

## 2:20–2:40 — Scorer
“Finally, I ran the supplied score.py. It validated all 12,000 IDs, positive prediction values, all 31 December dates and fixed scenario fields, and generated candidate_december.png.”

## 2:40–3:00 — Repository
“The repository contains the training script, scorer, dependencies, prediction output, December inputs, model artifact, report, and run instructions.”
