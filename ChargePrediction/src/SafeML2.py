import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import xlwings as xw
import os
from holidays import US as us_holidays
import traceback
import sys

import os
import sys

def check_environment():
    """Check if all requirements are met"""
    try:
        # Check Python version
        if sys.version_info < (3, 9):
            raise Exception("Python 3.9 or later is required")
        
        # Check Excel file exists
        excel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                "data", 
                                "LedgerBasedPredictionModeling.xlsm")
        if not os.path.exists(excel_path):
            raise Exception(f"Excel file not found at: {excel_path}")
            
        # Check required packages
        required_packages = ['pandas', 'numpy', 'statsmodels', 'sklearn', 
                           'xlwings', 'holidays']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                raise Exception(f"Required package '{package}' is not installed")
                
    except Exception as e:
        print(f"Environment Check Failed: {str(e)}")
        print("\nPlease ensure:")
        print("1. Python 3.9+ is installed")
        print("2. All required packages are installed (run install.bat)")
        print("3. Excel file is in the correct location")
        input("\nPress Enter to exit...")
        sys.exit(1)

class ChargePredictor:
    def __init__(self):
        try:
            excel_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    "data", 
                                    "LedgerBasedPredictionModeling.xlsm")
            self.wb = xw.Book(excel_path)
            
            # Get the 'Combine' sheet
            combine_sheet = self.wb.sheets['Combine']
            
            # Get the date from AvA sheet
            ava_sheet = self.wb.sheets['AvA']
            self.target_date = pd.to_datetime(ava_sheet.range('shiftDate').value)
            print(f"Target date from AvA sheet: {self.target_date}")
            
            # Read data from the Combine sheet
            data = combine_sheet.used_range.value
            
            # Convert to DataFrame
            headers = data[0]  # First row as headers
            self.df = pd.DataFrame(data[1:], columns=headers)
            
            # Print debug information
            print("\nOriginal columns:", self.df.columns.tolist())
            
            # Check if we have the required columns or need to select specific ones
            if len(self.df.columns) > 2:
                # Assuming the time and workable columns are the first two columns
                # Adjust these column indices if they're different in your Excel file
                self.df = self.df.iloc[:, [0, 1]]  # Select only the first two columns
            
            # Rename columns
            self.df.columns = ['Time', 'Workable']
            
            # Convert Time column to datetime
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            
            print("\nDataset Information:")
            print(f"Total rows: {len(self.df)}")
            print(f"Date range: {self.df['Time'].min()} to {self.df['Time'].max()}")
            
            # Get or create 'Collated' sheet
            try:
                self.collated_sheet = self.wb.sheets['Collated']
            except:
                self.collated_sheet = self.wb.sheets.add('Collated')
            
            self.prepare_data()
            self.train_sarima_model()
            self.train_rf_model()
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            print("\nDataFrame shape:", self.df.shape if hasattr(self, 'df') else "Not created")
            print("DataFrame columns:", self.df.columns.tolist() if hasattr(self, 'df') else "Not created")
            traceback.print_exc()
            raise


    def prepare_data(self):
        """Prepare and clean the data"""
        # Time features for SARIMA
        self.df['Hour'] = self.df['Time'].dt.hour
        self.df['DayOfWeek'] = self.df['Time'].dt.day_name()
        self.df['Month'] = self.df['Time'].dt.month
        self.df['Date'] = self.df['Time'].dt.date
        
        # Additional features for RF
        self.df['WeekOfYear'] = self.df['Time'].dt.isocalendar().week
        self.df['IsWeekend'] = self.df['Time'].dt.weekday >= 5
        self.df['IsHoliday'] = self.df['Time'].dt.date.apply(lambda x: x in us_holidays())
        self.df['IsNearHoliday'] = self.df.apply(self._is_near_holiday, axis=1)
        self.df['Season'] = self.df['Month'].apply(self._get_season)
        self.df['IsPeakHour'] = self.df['Hour'].apply(
            lambda x: 1 if (9 <= x <= 17) or (6 <= x <= 8) else 0
        )
        
        # Clean data
        self.df = self.df.dropna(subset=['Workable'])
        self.df = self.df.sort_values('Time')
        
        print(f"Total rows after cleaning: {len(self.df)}")

    def _is_near_holiday(self, row):
        date = row['Time'].date()
        for holiday_date in us_holidays().keys():
            if abs((date - holiday_date).days) <= 3:
                return True
        return False

    def _get_season(self, month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'

    # Keep existing SARIMA methods exactly as they are
    def train_sarima_model(self):
        """Train the SARIMA model"""
        try:
            # Prepare time series data
            self.ts_data = self.df.set_index('Time')['Workable']
            self.ts_data = self.ts_data.resample('h').mean()
            self.ts_data = self.ts_data.ffill()
            
            # Get the network final prediction from AvA sheet
            ava_sheet = self.wb.sheets['AvA']
            self.network_prediction = float(ava_sheet.range('INCRAMENT').value)
            print(f"\nNetwork Final Prediction (23:00): {self.network_prediction}")
    
            # Define SARIMA parameters
            self.sarima_model = SARIMAX(self.ts_data,
                                      order=(1, 1, 1),
                                      seasonal_order=(1, 1, 1, 24),
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)
            
            print("\nFitting SARIMA model...")
            self.sarima_results = self.sarima_model.fit(disp=False)
            print("SARIMA Model training completed")
            
        except Exception as e:
            print(f"Error in train_sarima_model: {str(e)}")

    def train_rf_model(self):
        """Train the Random Forest model"""
        try:
            # Prepare features for RF
            self.rf_features = ['Hour', 'Month', 'WeekOfYear', 'IsWeekend', 
                            'IsHoliday', 'IsNearHoliday', 'IsPeakHour']
            
            # One-hot encode categorical variables
            categorical_features = ['DayOfWeek', 'Season']
            
            # Store the unique categories for future reference
            self.day_categories = sorted(self.df['DayOfWeek'].unique())
            self.season_categories = sorted(self.df['Season'].unique())
            
            # Create and fit the encoder with feature names
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoder.fit(self.df[categorical_features])
            
            # Get encoded feature names for reference
            self.encoded_feature_names = self.encoder.get_feature_names_out(categorical_features)
            
            # Create training data
            X_categorical = self.encoder.transform(self.df[categorical_features])
            X_numerical = self.df[self.rf_features].values
            X = np.hstack([X_numerical, X_categorical])
            y = self.df['Workable'].values
            
            # Train Random Forest model
            self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.rf_model.fit(X, y)
            print("Random Forest Model training completed")
            print(f"Number of features used: {X.shape[1]}")
            
        except Exception as e:
            print(f"Error in train_rf_model: {str(e)}")
            traceback.print_exc()

    def get_rf_predictions(self):
        """Get Random Forest predictions"""
        try:
            pred_range = pd.date_range(
                start=self.target_date.replace(hour=0),
                periods=24,
                freq='h'
            )
            
            # Create a DataFrame for all predictions
            prediction_data = []
            for dt in pred_range:
                # Create a dictionary of features
                features_dict = {
                    'Hour': dt.hour,
                    'Month': dt.month,
                    'WeekOfYear': dt.isocalendar().week,
                    'IsWeekend': 1 if dt.weekday() >= 5 else 0,
                    'IsHoliday': 1 if dt.date() in us_holidays() else 0,
                    'IsNearHoliday': 1 if self._is_near_holiday({'Time': dt}) else 0,
                    'IsPeakHour': 1 if dt.hour in [9, 10, 11, 13, 14, 15] else 0,
                    'DayOfWeek': dt.day_name(),
                    'Season': self._get_season(dt.month)
                }
                prediction_data.append(features_dict)
            
            # Convert to DataFrame
            pred_df = pd.DataFrame(prediction_data)
            
            # Separate numerical and categorical features
            X_numerical = pred_df[self.rf_features].values
            X_categorical = self.encoder.transform(pred_df[['DayOfWeek', 'Season']])
            
            # Combine features
            X = np.hstack([X_numerical, X_categorical])
            
            # Make predictions
            predictions = self.rf_model.predict(X)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Time': pred_range.strftime('%Y-%m-%dT%H:00'),
                'RF_Prediction': np.round(np.maximum(predictions, 0))
            })
            
            return results_df
            
        except Exception as e:
            print(f"Error in get_rf_predictions: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def predict_for_target_date(self):
        """Make predictions for the target date with network prediction influence and same-day data"""
        try:
            # Create prediction range for the target date
            pred_range = pd.date_range(
                start=self.target_date.replace(hour=0),
                periods=24,
                freq='h'  # Changed from 'H' to 'h'
            )
            
            # Get same-day data
            same_day_data = self.df[self.df['Date'] == self.target_date.date()]
            last_known_hour = same_day_data['Hour'].max() if not same_day_data.empty else -1
            
            # Get base SARIMA predictions
            base_predictions = self.sarima_results.predict(
                start=pred_range[0],
                end=pred_range[-1]
            )
            
            # Calculate scaling factor based on network prediction
            scaling_factor = self.network_prediction / base_predictions.iloc[-1]
            
            # Apply weighted adjustment with same-day data influence
            adjusted_predictions = []
            
            for hour, base_pred in enumerate(base_predictions):
                if hour <= last_known_hour:
                    # Use actual data for known hours
                    adjusted_pred = same_day_data[same_day_data['Hour'] == hour]['Workable'].values[0]
                else:
                    if hour < 12:  # Early hours
                        weight_sarima = 0.60
                        weight_network = 0.20
                        weight_same_day = 0.20
                    elif hour < 18:  # Mid hours
                        weight_sarima = 0.50
                        weight_network = 0.30
                        weight_same_day = 0.20
                    else:  # Late hours
                        weight_sarima = 0.40
                        weight_network = 0.40
                        weight_same_day = 0.20
                    
                    # Calculate same-day trend
                    if last_known_hour >= 0:
                        same_day_trend = same_day_data['Workable'].diff().mean()
                    else:
                        same_day_trend = 0
                    
                    adjusted_pred = (base_pred * weight_sarima) + \
                                    (base_pred * scaling_factor * weight_network) + \
                                    (base_pred + same_day_trend * (hour - last_known_hour)) * weight_same_day
                
                adjusted_predictions.append(adjusted_pred)
            
            # Ensure predictions are non-negative and round to integers
            adjusted_predictions = np.maximum(adjusted_predictions, 0)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Time': pred_range,
                'Predicted_Workable': np.round(adjusted_predictions)
            })
            
            # Format the Time column
            results_df['Time'] = results_df['Time'].dt.strftime('%Y-%m-%dT%H:00')
            
            # Print weighting information
            print("\nPrediction Weighting Information:")
            print("Early hours (00:00-11:00): 60% SARIMA, 20% Network-influenced, 20% Same-day trend")
            print("Mid hours (12:00-17:00): 50% SARIMA, 30% Network-influenced, 20% Same-day trend")
            print("Late hours (18:00-23:00): 40% SARIMA, 40% Network-influenced, 20% Same-day trend")
            print(f"Network Prediction Scaling Factor: {scaling_factor:.2f}")
            print(f"Last known hour from same-day data: {last_known_hour}")
            
            return results_df
            
        except Exception as e:
            print(f"Error in predict_for_target_date: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()


    def write_predictions_to_excel(self, sarima_df, rf_df):
        """Write both predictions to the Collated sheet"""
        try:
            # Clear existing content
            self.collated_sheet.clear_contents()
            
            # Debug prints
            print("\nDebug Information:")
            print("SARIMA DataFrame:")
            print(sarima_df.columns.tolist())
            print(sarima_df.head())
            
            print("\nRF DataFrame:")
            print(rf_df.columns.tolist())
            print(rf_df.head())
            
            # Ensure proper time format for merging
            sarima_df['Time'] = pd.to_datetime(sarima_df['Time']).dt.strftime('%Y-%m-%dT%H:00')
            rf_df['Time'] = pd.to_datetime(rf_df['Time']).dt.strftime('%Y-%m-%dT%H:00')
            
            # Combine predictions
            combined_df = pd.merge(
                sarima_df, 
                rf_df[['Time', 'RF_Prediction']], 
                on='Time', 
                how='left'
            )
            
            print("\nCombined DataFrame:")
            print(combined_df.columns.tolist())
            print(combined_df.head())
            
            # Write headers
            headers = combined_df.columns.tolist()
            print("\nWriting headers:", headers)
            self.collated_sheet.range('A1').value = headers
            
            # Convert to list of lists for Excel
            data_to_write = combined_df.values.tolist()
            print(f"\nNumber of rows to write: {len(data_to_write)}")
            print(f"Number of columns to write: {len(data_to_write[0]) if data_to_write else 0}")
            
            # Write data
            if data_to_write:
                self.collated_sheet.range('A2').value = data_to_write
            
            # Auto-fit columns
            self.collated_sheet.autofit()
            
            print("Predictions written to Collated sheet successfully")
            
        except Exception as e:
            print(f"Error writing to Excel: {str(e)}")
            traceback.print_exc()

def main():
    check_environment()
    try:
        predictor = ChargePredictor()
        
        # Generate SARIMA predictions
        print("\nGenerating SARIMA predictions...")
        sarima_results = predictor.predict_for_target_date()
        print(f"SARIMA predictions shape: {sarima_results.shape}")
        
        # Generate RF predictions
        print("\nGenerating RF predictions...")
        rf_results = predictor.get_rf_predictions()
        print(f"RF predictions shape: {rf_results.shape}")
        
        # Write both results to Excel
        print("\nWriting to Excel...")
        predictor.write_predictions_to_excel(sarima_results, rf_results)
        
        print("\nPrediction completed successfully!")
        
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        traceback.print_exc()
    finally:
        # Always show this message and pause
        print("\nPress Enter to close this window...")
        input()

if __name__ == "__main__":
    main()
