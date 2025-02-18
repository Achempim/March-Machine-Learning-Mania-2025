# March-Machine-Learning-Mania-2025
This repository contains code for predicting outcomes of NCAA tournament games (both Men's and Women's) using logistic regression. The project includes data preprocessing, feature engineering, model training, evaluation, visualization, and prediction generation for all possible matchups.

- Table of Contents
- Overview
- Project Structure
- Installation & Requirements
- Usage
- Code Breakdown
- Visualizations
- Contributing
- License

## Overview
The goals of this project are to:

- Load and preprocess tournament data: Data includes tournament seeds, game results, team information, and conference data.
- Feature Engineering: Create features such as seed difference, score difference, and rank difference.
- Train Logistic Regression Models: Build separate models for the men’s and women’s tournaments.
- Evaluate Models: Compute metrics like accuracy, ROC AUC, log loss, and Brier score. Visualizations include ROC curves, confusion matrices, feature coefficients, and calibration curves.
- Generate Predictions: Predict game outcomes for all possible matchups, and combine these into a final submission file; and 
- Visualize Data Distributions: Plot histograms and distributions for both input features and prediction probabilities.

### Project Structure
<img width="313" alt="image" src="https://github.com/user-attachments/assets/84c5b2de-3522-4331-b401-7cc60d8b07c5" />

├── README.md
├── main.py                  # Main script containing data loading, model training, predictions, and visualizations
├── files/
│   ├── MNCAATourneySeeds.csv
│   ├── MRegularSeasonDetailedResults.csv
│   ├── MTeams.csv
│   ├── MTeamConferences.csv
│   ├── WNCAATourneySeeds.csv
│   ├── WRegularSeasonDetailedResults.csv
│   ├── WTeams.csv
│   └── WTeamConferences.csv
├── requirements.txt         # (Optional) Python dependencies
└── outputs/
    ├── men_roc_curve.png
    ├── women_roc_curve.png
    ├── men_confusion_matrix.png
    ├── women_confusion_matrix.png
    ├── men_feature_coefficients.png
    ├── women_feature_coefficients.png
    ├── men_calibration_curve.png
    ├── women_calibration_curve.png
    ├── march_mania_2025_men_submission.csv
    ├── march_mania_2025_women_submission.csv
    └── march_mania_2025_final_submission.csv

### Installation & Requirements
Ensure you have Python 3.6 or later installed. Install the required libraries using:

bash
pip install -r requirements.txt
If you don't have a requirements.txt, you can install the following packages manually:

pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

For example:

bash

pip install pandas numpy scikit-learn matplotlib seaborn joblib
Usage
Prepare your data:
Place all CSV files (e.g., tournament seeds, game results, teams, conferences) in the files/ directory.

Run the main script:
Execute the script in your terminal or command prompt:

bash

python main.py

This will:

- Load and preprocess the datasets.
- Perform feature engineering.
- Train logistic regression models for both men's and women's tournaments.
- Generate performance visualizations (ROC curves, confusion matrices, calibration curves, etc.).
- Generate predictions for all possible matchups in parallel.
- Save a final combined submission CSV file (march_mania_2025_final_submission.csv).
- Review Outputs:
- All visualizations are saved as PNG files, and predictions are saved in CSV format under the specified file names.

Code Breakdown
Data Loading & Preparation:
Loads tournament data from CSV files and merges multiple datasets (seeds, teams, conferences) to prepare a combined dataset.

Feature Engineering:
Extracts numerical seed values and computes differences (seed, score, rank) to be used as features.

Model Training & Evaluation:
Uses logistic regression (with balanced class weights) to train separate models for the men’s and women’s datasets. Evaluates the model with various metrics and produces visualizations (ROC curve, confusion matrix, etc.).

Prediction Generation:
Generates predictions for every possible matchup using a parallelized approach. The final predictions are combined into one CSV file.

Additional Visualizations:
Visualizes the distribution of engineered features, target labels, and predicted probabilities.

Visualizations
The following visualizations are generated:

ROC Curves: Evaluate the trade-off between true positive and false positive rates.
Confusion Matrices: Provide insight into model misclassifications.
Feature Coefficients: Show the importance and influence of each feature.
Calibration Curves: Compare predicted probabilities against actual outcomes.
Data Distributions: Histograms (with KDE) of feature distributions and target variable.
Predicted Probabilities Distribution: Visualizes how predictions are spread out.
Contributing
Contributions are welcome! Please fork this repository and submit a pull request with any improvements or bug fixes.

License
This project is licensed under the MIT License.
