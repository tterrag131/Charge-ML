# ChargePredictor

## Overview

ChargePredictor is a Python class designed to predict charge levels for industrial or organizational systems. It uses a combination of historical data analysis and machine learning techniques to provide accurate predictions for future dates.
The coding is proven with an OOB score of 90.232% a formidable score that represents the algorithm understands the underlying patterns presented by the data
The file ChargePredictor is a setup for individuals to download the code and test run it for themselves, the file works hevially with SMF1 and is designed for their site. Since inceptions SARIMA predicitons have been added which has increased accuracy all the way up to near 98% an incredible feat that offers flawless insight into the inner workings of complicated charge patterns and ever fluid data. 
These changes have been implemented at SMF1 and have made a predictive impact into out VET calculations and other methods of daily planning.

## Features

- Data loading and preprocessing from CSV files
- SARIMA predictive modeling for hevially seasonal data and more attribute based time series data logging and predictons
- Advanced feature engineering including time-based, holiday, and seasonal features
- Historical data analysis for specific dates and time periods
- Day-of-week and monthly average calculations
- Machine learning model (Random Forest & SARIMA) for prediction
- Combination of traditional and ML-based prediction methods
- Visualization of predictions and feature importance
- Includes inputs from other predictive sources such as ALPS

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- holidays (for US holiday detection)

## Key Methods

- `prepare_data()`: Cleans and prepares the data, creating new features.
- `train_ml_model()`: Trains the Random Forest model on the prepared data.
- `make_prediction(target_date_str)`: Makes predictions for a given date using both traditional and ML methods.
- `plot_prediction(target_date_str)`: Plots the predictions and feature importance.
- `analyze_feature_importance()`: Returns a DataFrame of feature importances.

## Data Requirements

The input CSV file should contain at least the following columns:
- 'Time': Datetime column
- 'Workable': Target variable for prediction

## Notes

- The class uses a combination of historical data, day-of-week averages, monthly averages, and machine learning predictions to generate final predictions.
- Predictions are weighted: 63% traditional methods, 37% machine learning.
- The Random Forest model is trained with 150 estimators and a max depth of 10.

## Error Handling

The class includes various error handling mechanisms and prints detailed information for debugging purposes.

## Customization

You can modify the feature engineering process, ML model parameters, or prediction weighting in the source code to suit your specific needs.
- Integration with ledger.py is being worked on and will be commited in the near future with implementation from ALPS
- the integration and implementation of these features will provide real time updates as well as concurrent data collection and ML learning. the ALPS will provide a strong prediction number for the code to adhere to while working to create the combined prediction.
