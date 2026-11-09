# Optiver Trading at the Close: Forecasting Closing Auction Price Moves

**Timeline:** 2025, revised November 2026 (rebuilt with out-of-time validation)

A LightGBM model that forecasts 60-second stock price moves during the Nasdaq closing auction, using order book and auction data from Kaggle's *Optiver Trading at the Close* competition (about 200 stocks over 481 trading days, 5.2 million rows). Completed independently after the competition closed.

## Highlights

- **Order book features:** bid-ask spread, auction imbalance ratio, and order flow (bid size minus ask size)
- **Out-of-time validation:** trained on the first 70% of trading days, used the next 10% only for early stopping, and scored once on the final 20%
- **No leakage from identifiers:** `date_id` and `time_id` are excluded so the model cannot memorize specific days
- **Honest benchmark:** compared against a naive model that always predicts zero

## Results (final 20% of trading days, never seen in training)

| Model | MAE (bps) |
|---|---|
| Always predict 0 | 6.062 |
| LightGBM | 5.971 |

LightGBM improves on the zero baseline by **1.5%** out of sample. That is a small edge, which is expected: 60-second price moves are close to unpredictable, so a modest but real out-of-sample improvement is the realistic goal.

**Most important features (gain):** order flow (24%) and spread (12%), followed by time within the auction and matched size.

## What I learned

An earlier version of this project used a random train/validation split. Because rows from the same trading day landed in both sets, the model was partly tested on days it had already seen, and `date_id` showed up as one of the most important features. Switching to an out-of-time split and removing identifier columns gave a lower but trustworthy result.

## Next steps

- Rolling walk-forward validation across several periods instead of one split
- Features capturing how the auction evolves from one snapshot to the next, and how each stock moves relative to others at the same time
- Hyperparameter tuning on the walk-forward folds

## How to run

The easiest way is on Kaggle: create a notebook in the [Optiver Trading at the Close competition](https://www.kaggle.com/competitions/optiver-trading-at-the-close), import `optiver_closing_auction.ipynb`, and run all cells. To run locally, download `train.csv` from the competition and update the path in the first code cell. The data is not included here because of Kaggle's competition rules.

## Tools

Python, pandas, NumPy, LightGBM, scikit-learn, Matplotlib
