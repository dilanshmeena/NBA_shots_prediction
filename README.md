# 🏀 NBA Shot Selection Prediction (PRCP-1008)

Predict whether an NBA shot is made or missed, based on the conditions it was taken in.

## Problem Statement
1. Prepare a complete data analysis report on the given data.
2. Help stakeholders develop better game strategies by finding the model with the highest accuracy and F1-score.
3. Build a shot prediction model that predicts whether a shot is scored, based on the circumstances of the shot.

## Dataset
- **Size:** 30,697 shots, 25,697 with a known outcome and 5,000 with a missing outcome (`shot_made_flag`).
- **Target:** `shot_made_flag` (1 = made, 0 = missed).
- **Main columns:** court location (`loc_x`, `loc_y`), `shot_distance`, time left, `period`, `playoffs`, shot type and shot zone columns, `matchup`, `season`.
- The 5,000 shots with no outcome are predicted at the end of the notebook.

## Approach
1. **Cleaning:** dropped redundant or useless columns (`lat`, `lon`, team and game IDs) and removed the top 2% outliers with Isolation Forest.
2. **Feature engineering:** created `time_remaining` (seconds left), `is_home` (from `matchup`) and `season_num` (from `season`).
3. **EDA:** made vs missed shots, accuracy by distance, 2PT vs 3PT, shot locations on the court, and home/away and playoff splits.
4. **Leak-free preprocessing:** split the data first, then fitted the smoothed target encoding and the standard scaler on the training rows only.
5. **Modeling:** Logistic Regression, Random Forest (base and tuned with GridSearchCV) and Gradient Boosting.
6. **Validation:** a held-out validation set plus 5-fold cross-validation.
7. **Final model:** refit on all labeled shots to predict the 5,000 missing shots.

## Key Findings
- Overall field goal percentage is about 45%.
- 2-point shots go in about 47.7% of the time, and 3-point shots about 34.0%.
- Make percentage falls as the shot gets further from the hoop.

## Results

Validation set:

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Gradient Boosting | 0.6831 | 0.5682 | 0.7363 | 0.4626 | 0.7033 |
| **Logistic Regression** | 0.6824 | **0.5740** | 0.7254 | 0.4749 | 0.6996 |
| Random Forest (Tuned) | 0.6804 | 0.5674 | 0.7273 | 0.4652 | 0.7035 |
| Random Forest (Base) | 0.6512 | 0.5637 | 0.6460 | 0.5000 | 0.6712 |

5-fold cross-validation (training data):

| Model | CV Accuracy | CV F1 |
|---|---|---|
| Logistic Regression | 0.6787 ± 0.0108 | 0.5666 ± 0.0141 |
| Random Forest (Tuned) | 0.6793 ± 0.0101 | 0.5637 ± 0.0139 |

## Conclusion
- The top three models are within 0.3 percentage points of each other in accuracy, so there is no real winner on accuracy.
- **Logistic Regression** is the final model. It has the best F1-score and is the simplest and easiest to explain.
- Precision is about 0.73 but recall is about 0.47, so the model is right most of the time when it predicts a make, but it misses more than half of the actual makes.
- Shot outcomes are quite random, so about 68% accuracy is reasonable for the features available.

## Project Structure
```
NBA-Shot-Selection/
├── NBA_shots_prediction.ipynb   # full analysis and models
├── NBAShotSelection.csv         # dataset
├── NBA_shot_predictions.csv     # predictions for the 5,000 missing shots
├── requirements.txt
└── README.md
```

## How to Run
```bash
git clone https://github.com/<your-username>/NBA-Shot-Selection.git
cd NBA-Shot-Selection
pip install -r requirements.txt
jupyter notebook NBA_shots_prediction.ipynb
```
Run all cells. The grid search takes a few minutes.

## Tech Stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook.

## Possible Improvements
- Add player-level or defender features, if available.
- Try other models such as XGBoost or LightGBM.
- Tune the decision threshold to improve recall.
