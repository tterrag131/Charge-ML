import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import xlwings as xw
import os
from holidays import US as us_holidays
import traceback
import sys
import signal
import json
import boto3
import io
import schedule
import time
from prophet import Prophet
from datetime import datetime, timedelta
from botocore.exceptions import NoCredentialsError

S3_BUCKET_NAME = None

def run_scheduled_task():
    try:
        print(f"\nStarting scheduled run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        main()  # Your existing main function
        print(f"Completed run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"Error in scheduled run: {str(e)}")
        traceback.print_exc()

def check_environment():
    """Check if all requirements are met"""
    try:
        # Check Python version
        if sys.version_info < (3, 9):
            raise Exception("Python 3.9 or later is required")
            
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
            # Initialize S3 client
            self.s3 = boto3.client('s3', region_name='us-west-1')
            self.bucket_name = 'ledger-prediction-charting-008971633421'

            # Get combine.csv from S3
            try:
                print("Fetching combine.csv from S3...")
                combine_obj = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key='userloads/combine.csv'
                )
                combine_data = combine_obj['Body'].read().decode('utf-8')
                self.df = pd.read_csv(io.StringIO(combine_data))
                print("Successfully loaded combine.csv from S3")
            except Exception as e:
                print(f"Error reading combine.csv from S3: {str(e)}")
                raise
            
            # Get the date from user input
            self.target_date = pd.to_datetime('today').normalize()
            print(f"Target date: {self.target_date.strftime('%m/%d/%Y')}")
            
            # Print debug information
            print("\nOriginal columns:", self.df.columns.tolist())
            
            # Check reqs
            if len(self.df.columns) > 2:
                self.df = self.df.iloc[:, [0, 1]]  # Select he first two columns
            
            # Rename columns
            self.df.columns = ['Time', 'Workable']
            
            # Convert Time column to datetime
            self.df['Time'] = pd.to_datetime(self.df['Time'])
            
            print("\nDataset Information:")
            print(f"Total rows: {len(self.df)}")
            print(f"Date range: {self.df['Time'].min()} to {self.df['Time'].max()}")
            
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
        """Train the Prophet model (formerly SARIMA)"""
        try:
            # Prepare time series data for Prophet
            # Prophet requires columns named 'ds' (datetime) and 'y' (value)
            self.ts_data = self.df.set_index('Time')['Workable']
            self.ts_data = self.ts_data.resample('h').mean()
            self.ts_data = self.ts_data.ffill()

            # Convert Series to DataFrame for Prophet
            prophet_df = self.ts_data.reset_index()
            prophet_df.columns = ['ds', 'y']
            
            # Get granular.csv from S3 and extract network prediction
            try:
                print("Fetching granular.csv from S3...")
                granular_obj = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key='userloads/granular.csv'
                )
                granular_data = granular_obj['Body'].read().decode('utf-8')
                granular_df = pd.read_csv(io.StringIO(granular_data))
                self.network_prediction = float(granular_df['IPTNW'].iloc[0])
                print(f"Network prediction loaded: {self.network_prediction}")
            except Exception as e:
                print(f"Error reading granular.csv from S3: {str(e)}")
                raise

            # Define Prophet model and parameters
            # Prophet automatically handles trend and seasonality.
            # We explicitly add daily seasonality since your data is hourly with a daily reset.
            # You can also add weekly_seasonality=True if that applies.
            self.prophet_model = Prophet(
                daily_seasonality=True,
                yearly_seasonality=False, # Assuming no yearly cycle is relevant here
                weekly_seasonality=False # Set to True if applicable
            )
            
            # You might want to tune these further:
            # seasonality_mode='multiplicative' if seasonality scales with magnitude
            # changepoint_prior_scale for trend flexibility (default is 0.05)

            print("\nFitting Prophet model...")
            self.prophet_results = self.prophet_model.fit(prophet_df)
            print("Prophet Model training completed")
            
        except Exception as e:
            print(f"Error in train_sarima_model (now Prophet): {str(e)}") # Updated print
            raise

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
            print("RF Model training completed")
            print(f"Number of features used: {X.shape[1]}")
            
        except Exception as e:
            print(f"Error in train_rf_model: {str(e)}")
            traceback.print_exc()

    def get_rf_predictions(self):
        """Get RF predictions"""
        try:
            pred_range = pd.date_range(
                start=self.target_date.replace(hour=0),
                periods=24,
                freq='h'
            )
            
            #DataFrame for all predictions
            prediction_data = []
            for dt in pred_range:
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

    def predict_rolling_48h(self):
        """Generate predictions for next 48 hours from given start time"""
        try:
            current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            pred_range = pd.date_range(
                start=current_time,
                periods=49,  # 49 to include both start and end hour
                freq='h'
            )
            
            predictions = []
            scaling_data = self.get_network_scaling_factors()
            
            for hour_idx, target_time in enumerate(pred_range[:-1]):
                # Force midnight predictions to 0
                if target_time.hour == 0:
                    predictions.append({
                        "Time": target_time.strftime('%Y-%m-%dT%H:00'),
                        "Predicted_Workable": 0
                    })
                    continue
                    
                # Get base SARIMA prediction using iloc for positional indexing
                base_pred = self.sarima_results.predict(
                    start=target_time,
                    end=target_time
                ).iloc[0]
                
                hour_in_day = target_time.hour
                day_offset = (target_time.date() - current_time.date()).days
                
                if day_offset == 0:
                    scaling_factor = scaling_data['hourly_factors'][hour_in_day]
                else:
                    scaling_factor = scaling_data['hourly_factors'][hour_in_day] * 0.95
                    
                adjusted_pred = max(0, round(base_pred * scaling_factor))
                
                predictions.append({
                    "Time": target_time.strftime('%Y-%m-%dT%H:00'),
                    "Predicted_Workable": adjusted_pred
                })
                
            return pd.DataFrame(predictions)
            
        except Exception as e:
            print(f"Error in rolling 48h prediction: {str(e)}")
            traceback.print_exc()
            return None
    
    def generate_rolling_predictions(self):
        """Generate and format all predictions into JSON structure"""
        try:
            current_time = datetime.now().replace(minute=0, second=0, microsecond=0)
            
            # Generate 48-hour rolling predictions
            rolling_predictions = self.predict_rolling_48h()
            
            if rolling_predictions is None:
                raise Exception("Failed to generate rolling predictions")
            
            # Split predictions into time windows
            current_day_end = current_time.replace(hour=23, minute=59, second=59)
            next_day_start = (current_time + timedelta(days=1)).replace(hour=0, minute=0, second=0)
            next_day_end = next_day_start.replace(hour=23, minute=59, second=59)
            
            # Split predictions into different windows
            current_day_preds = rolling_predictions[
                rolling_predictions['Time'].apply(lambda x: pd.to_datetime(x) <= current_day_end)
            ]
            next_day_preds = rolling_predictions[
                (rolling_predictions['Time'].apply(lambda x: 
                    next_day_start <= pd.to_datetime(x) <= next_day_end))
            ]
            remainder_preds = rolling_predictions[
                rolling_predictions['Time'].apply(lambda x: pd.to_datetime(x) > next_day_end)
            ]
            
            # Get historical data
            prev_year_date = current_time - pd.DateOffset(years=1)
            prev_year_records = self.get_previous_year_data(prev_year_date)
            next_day_prev_year_records = self.get_previous_year_data(prev_year_date + timedelta(days=1))
            
            # Get current day actual data
            current_day_data = self.get_current_day_data(current_time)
            
            json_data = {
                "time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "current_day": {
                    "date": current_time.strftime("%Y-%m-%d"),
                    "network_prediction": self.network_prediction,
                    "sarima_predictions": current_day_preds.to_dict(orient='records'),
                    "rf_predictions": self.get_rf_predictions().to_dict(orient='records'),
                    "predictions_no_same_day": self.predict_without_same_day_influence(current_time).to_dict(orient='records'),
                    "previous_year_data": prev_year_records,
                    "current_day_data": current_day_data
                },
                "next_day": {
                    "date": next_day_start.strftime("%Y-%m-%d"),
                    "sarima_predictions": next_day_preds.to_dict(orient='records'),
                    "rf_predictions": self.get_rf_predictions().to_dict(orient='records'),
                    "previous_year_data": next_day_prev_year_records
                },
                "extended_predictions": {
                    "predictions": remainder_preds.to_dict(orient='records') if not remainder_preds.empty else []
                },
                "Ledger_Information": self.get_ledger_information() if hasattr(self, 'get_ledger_information') else {}
            }
            
            return json_data
            
        except Exception as e:
            print(f"Error in generate_rolling_predictions: {str(e)}")
            traceback.print_exc()
            return None
    
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
                        weight_sarima = 0.50
                        weight_network = 0.30
                        weight_same_day = 0.20
                    elif hour < 18:  # Mid hours
                        weight_sarima = 0.50
                        weight_network = 0.30
                        weight_same_day = 0.20
                    else:  # Late hours
                        weight_sarima = 0.40
                        weight_network = 0.35
                        weight_same_day = 0.25
                    
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
    
    def predict_without_same_day_influence(self, target_date):
        """Make predictions without same-day data influence"""
        try:
            # Create prediction range for the target date
            pred_range = pd.date_range(
                start=target_date.replace(hour=0),
                periods=24,
                freq='h'
            )
            
            # Get base SARIMA predictions
            base_predictions = self.sarima_results.predict(
                start=pred_range[0],
                end=pred_range[-1]
            )
            
            # Calculate scaling factor based on network prediction
            scaling_factor = self.network_prediction / base_predictions.iloc[-1]
            
            # Apply weighted adjustment without same-day influence
            adjusted_predictions = []
            
            for hour, base_pred in enumerate(base_predictions):
                if hour < 12:  # Early hours
                    weight_sarima = 0.70
                    weight_network = 0.30
                elif hour < 18:  # Mid hours
                    weight_sarima = 0.60
                    weight_network = 0.40
                else:  # Late hours
                    weight_sarima = 0.50
                    weight_network = 0.50
                
                adjusted_pred = (base_pred * weight_sarima) + \
                            (base_pred * scaling_factor * weight_network)
                
                adjusted_predictions.append(adjusted_pred)
            
            # Ensure predictions are non-negative and round to integers
            adjusted_predictions = np.maximum(adjusted_predictions, 0)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Time': pred_range,
                'Predicted_Workable_No_Same_Day': np.round(adjusted_predictions)
            })
            
            # Format the Time column
            results_df['Time'] = results_df['Time'].dt.strftime('%Y-%m-%dT%H:00')
            
            return results_df
            
        except Exception as e:
            print(f"Error in predict_without_same_day_influence: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_alps_data(self):
        """Retrieve and validate ALPS data"""
        try:
            alps_obj = self.s3.get_object(
                Bucket=self.bucket_name,
                Key='userloads/alps.csv'
            )
            alps_data = alps_obj['Body'].read().decode('utf-8')
            alps_df = pd.read_csv(io.StringIO(alps_data))
            
            # Print debug information
            print("\nOriginal ALPS columns:", alps_df.columns.tolist())
            print("Sample data:")
            print(alps_df[['Site', 'Shift', 'Value', 'Date']].head())
            
            # Filter for SMF1 data and most recent date
            alps_df = alps_df[alps_df['Site'] == 'SMF1']
            alps_df['Date'] = pd.to_datetime(alps_df['Date'])
            latest_date = alps_df['Date'].max()
            alps_df = alps_df[alps_df['Date'] == latest_date]
            
            # Get day and night shift values
            day_shift_value = alps_df[alps_df['Shift'] == 'DAY_SHIFT']['Value'].iloc[0] if not alps_df[alps_df['Shift'] == 'DAY_SHIFT'].empty else 0
            night_shift_value = alps_df[alps_df['Shift'] == 'NIGHT_SHIFT']['Value'].iloc[0] if not alps_df[alps_df['Shift'] == 'NIGHT_SHIFT'].empty else 0
            
            # Create formatted DataFrame
            formatted_alps = pd.DataFrame({
                'DAY_SHIFT': [day_shift_value],
                'NIGHT_SHIFT': [night_shift_value]
            })
            
            print("\nALPS Data Retrieved:")
            print(f"Date: {latest_date}")
            print(f"Day Shift: {day_shift_value:,.2f}")
            print(f"Night Shift: {night_shift_value:,.2f}")
            
            return formatted_alps
            
        except Exception as e:
            print(f"Error retrieving ALPS data: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_ledger_information(self):
        try:
            # Get granular.csv from S3
            print("Fetching granular.csv from S3 for ledger information...")
            try:
                granular_obj = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key='userloads/granular.csv'
                )
                granular_data = granular_obj['Body'].read().decode('utf-8')
                df = pd.read_csv(io.StringIO(granular_data))
                print("Successfully loaded granular.csv from S3")
            except Exception as e:
                print(f"Error reading granular.csv from S3: {str(e)}")
                raise

            # Get current hour
            current_hour = datetime.now().hour

            # Generate time points from 00:00 up to current hour
            formatted_times = [f"{hour:02d}:00" for hour in range(current_hour + 1)]

            # Create metrics dictionary
            metrics = {}
            for column in df.columns:
                if column != 'Hour':  # Skip the time column
                    # Get values and convert to appropriate numeric format
                    values = []
                    for v in df[column].values:
                        try:
                            if pd.notna(v) and str(v).strip():
                                num = float(v)
                                # Check if it's a whole number
                                if num.is_integer():
                                    values.append(int(num))
                                else:
                                    # Round to 2 decimal places
                                    values.append(round(num, 2))
                            else:
                                values.append(0)
                        except (ValueError, TypeError):
                            values.append(0)
                    metrics[column] = values[:current_hour + 1]  # Ensure we only take up to current hour

            # Create the ledger information structure
            ledger_info = {
                "timePoints": formatted_times,
                "metrics": metrics
            }

            # Add debug logging
            print(f"Ledger info created with {len(formatted_times)} time points")
            print(f"Metrics included: {list(metrics.keys())}")

            return ledger_info

        except Exception as e:
            print(f"Error processing ledger information: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_fallback_scaling_factors(self):
        """Provide fallback scaling if ALPS data is unavailable"""
        try:
            # Get network predictions from granular.csv
            granular_obj = self.s3.get_object(
                Bucket=self.bucket_name,
                Key='userloads/granular.csv'
            )
            granular_data = granular_obj['Body'].read().decode('utf-8')
            granular_df = pd.read_csv(io.StringIO(granular_data))
    
            # Get the most recent values
            current_target = float(granular_df['IPTNW'].iloc[0])
            three_day_target = float(granular_df['IPT3DAY'].iloc[0])
    
            # Use original scaling logic as fallback
            trend_factor = three_day_target / current_target
    
            if current_target < three_day_target:
                base_scaling = 1 + (trend_factor - 1) * 0.85
                next_day_scaling = min(1.6, base_scaling)
            else:
                base_scaling = 1 - (1 - trend_factor) * 0.85
                next_day_scaling = max(0.65, base_scaling)
    
            hourly_factors = {}
            for hour in range(24):
                if hour < 3:
                    hourly_factors[hour] = next_day_scaling * 0.87
                elif 3 <= hour < 6:
                    hourly_factors[hour] = next_day_scaling * 1
                elif 6 <= hour < 9:
                    hourly_factors[hour] = next_day_scaling * 1.03
                elif 9 <= hour < 12:
                    hourly_factors[hour] = next_day_scaling * 1.07
                elif 12 <= hour < 15:
                    hourly_factors[hour] = next_day_scaling * 1.08
                elif 15 <= hour < 18:
                    hourly_factors[hour] = next_day_scaling * 1.06
                elif 18 <= hour < 21:
                    hourly_factors[hour] = next_day_scaling * 1.01
                else:
                    hourly_factors[hour] = next_day_scaling * 0.95
                
    
            return {
                'base_scaling': base_scaling,
                'next_day_scaling': next_day_scaling,
                'hourly_factors': hourly_factors,
                'metrics': {
                    'current_target': current_target,
                    'three_day_target': three_day_target,
                    'trend_factor': trend_factor,
                    'avg_scaling': sum(hourly_factors.values()) / 24
                }
            }
    
        except Exception as e:
            print(f"Error in fallback scaling: {str(e)}")
            traceback.print_exc()
            return None
    
    def get_network_scaling_factors(self):
        """Calculate scaling factors using ATHENA, IPT, and ALPS data with weighted influence"""
        try:
            # Get ALPS data
            alps_df = self.get_alps_data()
            
            # Get network predictions from granular.csv
            granular_obj = self.s3.get_object(
                Bucket=self.bucket_name,
                Key='userloads/granular.csv'
            )
            granular_data = granular_obj['Body'].read().decode('utf-8')
            granular_df = pd.read_csv(io.StringIO(granular_data))
    
            # Get current targets
            current_ipt = float(granular_df['IPTNW'].iloc[0])
            three_day_ipt = float(granular_df['IPT3DAY'].iloc[0])
            
            if alps_df is None:
                # Fallback logic remains the same
                trend_factor = three_day_ipt / current_ipt
                
                if current_ipt < three_day_ipt:
                    base_scaling = 1 + (trend_factor - 1) * 0.85
                    next_day_scaling = min(1.6, base_scaling)
                else:
                    base_scaling = 1 - (1 - trend_factor) * 0.85
                    next_day_scaling = max(0.65, base_scaling)
            else:
                # Get total ALPS forecast
                current_alps_total = float(alps_df['DAY_SHIFT'].iloc[0] + alps_df['NIGHT_SHIFT'].iloc[0])
                
                # Calculate trend factors with adjusted weights
                ipt_trend = three_day_ipt / current_ipt
                alps_trend = current_alps_total / current_ipt
                
                # Adjust weights to give more influence to ALPS total forecast
                weighted_trend = (
                    (ipt_trend * 0.35) +      # IPT weight increased
                    (alps_trend * 0.35) +     # ALPS weight increased
                    (1.0 * 0.3)               # Base weight decreased
                )
    
                # Adjust scaling factors to be more aggressive
                if weighted_trend > 1:
                    base_scaling = 1 + (weighted_trend - 1) * 0.9  # Increased from 0.75
                    next_day_scaling = min(1.65, base_scaling)     # Increased cap
                else:
                    base_scaling = 1 - (1 - weighted_trend) * 0.9
                    next_day_scaling = max(0.75, base_scaling)     # Increased floor
    
            # Create hourly factors with enhanced scaling
            hourly_factors = {}
            for hour in range(24):
                base_factor = next_day_scaling
                
                # Adjust time-of-day factors to be more aggressive
                if hour < 6:  # Early morning
                    tod_factor = 0.99  # Increased from 0.92
                elif 6 <= hour < 12:  # Morning peak
                    tod_factor = 1.05  # Increased from 1.08
                elif 12 <= hour < 18:  # Afternoon
                    tod_factor = 1.03  # Increased from 1.05
                else:  # Evening
                    tod_factor = 1.03  # Increased from 0.95
    
                # If ALPS data is available, incorporate total volume distribution
                if alps_df is not None:
                    total_alps = float(alps_df['DAY_SHIFT'].iloc[0] + alps_df['NIGHT_SHIFT'].iloc[0])
                    day_ratio = float(alps_df['DAY_SHIFT'].iloc[0]) / total_alps
                    
                    # Apply time-based volume distribution
                    if 6 <= hour < 18:  # Day shift hours
                        volume_factor = 1 + (day_ratio * 0.15)  # Boost day shift slightly
                    else:
                        volume_factor = 1 + ((1 - day_ratio) * 0.15)  # Boost night shift slightly
                    
                    hourly_factors[hour] = base_factor * tod_factor * volume_factor
                else:
                    hourly_factors[hour] = base_factor * tod_factor
    
            # Calculate average scaling
            avg_scaling = sum(hourly_factors.values()) / 24
    
            # Print debug information
            print("\nScaling Factor Analysis:")
            print(f"IPT Trend: {ipt_trend:.3f}")
            if alps_df is not None:
                print(f"ALPS Trend: {alps_trend:.3f}")
                print(f"Weighted Trend: {weighted_trend:.3f}")
            print(f"Base Scaling: {base_scaling:.3f}")
            print(f"Next Day Scaling: {next_day_scaling:.3f}")
            print(f"Average Hourly Scaling: {avg_scaling:.3f}")
    
            return {
                'base_scaling': base_scaling,
                'next_day_scaling': next_day_scaling,
                'hourly_factors': hourly_factors,
                'metrics': {
                    'current_ipt': current_ipt,
                    'three_day_ipt': three_day_ipt,
                    'trend_factor': weighted_trend if alps_df is not None else trend_factor,
                    'avg_scaling': avg_scaling
                }
            }
    
        except Exception as e:
            print(f"Error in get_network_scaling_factors: {str(e)}")
            traceback.print_exc()
            return None
    
    def calculate_historical_trends(self, days_prior=45, num_weeks_for_avg=6, short_term_ma_occurrences=3):
        """
        Calculates historical trends based on the last 'days_prior' days of data,
        focusing on 3-hour blocks and daily summaries, including moving averages,
        a percentage change to indicate trend direction, and the last occurrence total for each day.
        """
        print(f"\nCalculating historical trends for the last {days_prior} days (long-term avg: {num_weeks_for_avg} occurrences, short-term MA: {short_term_ma_occurrences} occurrences)...")
        if not hasattr(self, 'df') or self.df.empty:
            print("Historical dataframe not available. Skipping trend calculation.")
            return {
                "reference_date_for_trends": self.target_date.strftime("%Y-%m-%d") if hasattr(self, 'target_date') else "N/A",
                "trend_period_days": days_prior,
                "message": "Historical dataframe not available or empty."
            }

        try:
            if not pd.api.types.is_datetime64_any_dtype(self.df['Time']):
                self.df['Time'] = pd.to_datetime(self.df['Time'])
            if not pd.api.types.is_numeric_dtype(self.df['Workable']):
                self.df['Workable'] = pd.to_numeric(self.df['Workable'], errors='coerce').fillna(0)

            end_date_for_trends = self.target_date - timedelta(days=1) 
            start_date_for_trends = end_date_for_trends - timedelta(days=days_prior - 1)

            print(f"Historical trend calculation range: {start_date_for_trends.strftime('%Y-%m-%d')} to {end_date_for_trends.strftime('%Y-%m-%d')}")

            hist_df = self.df[
                (self.df['Time'].dt.normalize() >= pd.to_datetime(start_date_for_trends)) &
                (self.df['Time'].dt.normalize() <= pd.to_datetime(end_date_for_trends))
            ].copy()

            if hist_df.empty:
                print(f"No historical data found in the last {days_prior} days. Skipping trend calculation.")
                return {
                    "reference_date_for_trends": self.target_date.strftime("%Y-%m-%d"),
                    "trend_period_days": days_prior,
                    "message": "No historical data in the specified period."
                }

            hist_df['DayName'] = hist_df['Time'].dt.day_name()
            hist_df['DateOnly'] = hist_df['Time'].dt.date
            if 'Hour' not in hist_df.columns:
                 hist_df['Hour'] = hist_df['Time'].dt.hour

            three_hour_blocks = [
                (0, 3, "0000_0300"), (3, 6, "0300_0600"), (6, 9, "0600_0900"),
                (9, 12, "0900_1200"), (12, 15, "1200_1500"), (15, 18, "1500_1800"),
                (18, 21, "1800_2100"), (21, 24, "2100_0000")
            ]

            block_trends = {}
            daily_summary = defaultdict(lambda: {}) 

            all_block_data_points = []
            for date_only_val in sorted(hist_df['DateOnly'].unique()):
                day_df = hist_df[hist_df['DateOnly'] == date_only_val].copy()
                day_df = day_df.set_index('Time').sort_index()
                
                for start_hour, end_hour, block_label_suffix in three_hour_blocks:
                    vol_at_block_start = 0
                    if start_hour > 0:
                        start_lookup_timestamp = pd.Timestamp(date_only_val).replace(hour=start_hour - 1, minute=59, second=59, microsecond=999999)
                        temp_df_start = day_df.loc[day_df.index <= start_lookup_timestamp]
                        if not temp_df_start.empty:
                            vol_at_block_start = temp_df_start['Workable'].iloc[-1]
                    
                    actual_end_hour_for_lookup = end_hour - 1
                    if end_hour == 24: actual_end_hour_for_lookup = 23
                    
                    end_lookup_timestamp = pd.Timestamp(date_only_val).replace(hour=actual_end_hour_for_lookup, minute=59, second=59, microsecond=999999)
                    temp_df_end = day_df.loc[day_df.index <= end_lookup_timestamp]
                    vol_at_block_end = vol_at_block_start 
                    if not temp_df_end.empty:
                        vol_at_block_end = temp_df_end['Workable'].iloc[-1]
                                        
                    block_volume = max(0, vol_at_block_end - vol_at_block_start)
                    
                    all_block_data_points.append({
                        'DayName': pd.Timestamp(date_only_val).day_name(),
                        'BlockLabelSuffix': block_label_suffix,
                        'DateOnly': date_only_val,
                        'BlockVolume': block_volume
                    })

            block_data_df = pd.DataFrame(all_block_data_points)
            days_of_week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            if not block_data_df.empty:
                for day_name in days_of_week_order:
                    for _, _, block_label_suffix in three_hour_blocks:
                        block_key = f"{day_name}_{block_label_suffix}"
                        specific_block_occurrences = block_data_df[
                            (block_data_df['DayName'] == day_name) &
                            (block_data_df['BlockLabelSuffix'] == block_label_suffix)
                        ].sort_values(by='DateOnly', ascending=False)

                        avg_vol_long, avg_vol_short, last_occurrence_vol, trend_pct_change = 0, 0, 0, 0.0
                        if not specific_block_occurrences.empty:
                            long_term_avg_occurrences = specific_block_occurrences.head(num_weeks_for_avg)
                            short_term_avg_occurrences = specific_block_occurrences.head(short_term_ma_occurrences)
                            avg_vol_long = long_term_avg_occurrences['BlockVolume'].mean() if not long_term_avg_occurrences.empty else 0
                            avg_vol_short = short_term_avg_occurrences['BlockVolume'].mean() if not short_term_avg_occurrences.empty else 0
                            last_occurrence_vol = specific_block_occurrences.iloc[0]['BlockVolume']
                            if avg_vol_long != 0: trend_pct_change = ((avg_vol_short - avg_vol_long) / avg_vol_long) * 100
                            elif avg_vol_short > 0 : trend_pct_change = 9999.0
                        
                        block_trends[block_key] = {
                            f"avg_volume_last_{num_weeks_for_avg}_occurrences": round(avg_vol_long, 2),
                            f"avg_volume_last_{short_term_ma_occurrences}_occurrences": round(avg_vol_short, 2),
                            "last_occurrence_volume": round(last_occurrence_vol, 2),
                            "trend_direction_pct_change": round(trend_pct_change, 2)
                        }
            else: 
                for day_name_iter in days_of_week_order:
                     for _, _, block_label_suffix_iter in three_hour_blocks:
                        block_key_iter = f"{day_name_iter}_{block_label_suffix_iter}"
                        block_trends[block_key_iter] = {
                            f"avg_volume_last_{num_weeks_for_avg}_occurrences": 0,
                            f"avg_volume_last_{short_term_ma_occurrences}_occurrences": 0,
                            "last_occurrence_volume": 0,
                            "trend_direction_pct_change": 0.0
                        }

            daily_totals_df = pd.DataFrame()
            if not block_data_df.empty:
                daily_totals_df = block_data_df.groupby(['DateOnly', 'DayName'])['BlockVolume'].sum().reset_index()
            
            for day_name_iter in days_of_week_order:
                daily_summary[day_name_iter] = {
                    f"avg_total_daily_volume_last_{num_weeks_for_avg}_occurrences": 0,
                    f"avg_total_daily_volume_last_{short_term_ma_occurrences}_occurrences": 0,
                    "last_occurrence_total_daily_volume": 0, # NEW: Initialize
                    "trend_direction_pct_change": 0.0
                }

            if not daily_totals_df.empty:
                for day_name in daily_totals_df['DayName'].unique(): # Iterate only days present in data
                    day_specific_totals = daily_totals_df[daily_totals_df['DayName'] == day_name].sort_values(by='DateOnly', ascending=False)
                    if not day_specific_totals.empty:
                        long_term_daily_totals = day_specific_totals.head(num_weeks_for_avg)
                        short_term_daily_totals = day_specific_totals.head(short_term_ma_occurrences)
                        
                        avg_daily_total_long = long_term_daily_totals['BlockVolume'].mean() if not long_term_daily_totals.empty else 0
                        avg_daily_total_short = short_term_daily_totals['BlockVolume'].mean() if not short_term_daily_totals.empty else 0
                        last_occurrence_daily_total = day_specific_totals.iloc[0]['BlockVolume'] # NEW: Get last occurrence total

                        daily_trend_pct_change = 0.0
                        if avg_daily_total_long != 0:
                            daily_trend_pct_change = ((avg_daily_total_short - avg_daily_total_long) / avg_daily_total_long) * 100
                        elif avg_daily_total_short > 0:
                            daily_trend_pct_change = 9999.0
                        
                        daily_summary[day_name].update({
                            f"avg_total_daily_volume_last_{num_weeks_for_avg}_occurrences": round(avg_daily_total_long, 2),
                            f"avg_total_daily_volume_last_{short_term_ma_occurrences}_occurrences": round(avg_daily_total_short, 2),
                            "last_occurrence_total_daily_volume": round(last_occurrence_daily_total, 2), # NEW
                            "trend_direction_pct_change": round(daily_trend_pct_change, 2)
                        })
            
            overall_avg_daily_45_days = daily_totals_df['BlockVolume'].mean() if not daily_totals_df.empty else 0
            avg_daily_volume_rolling_7_days = 0
            if not daily_totals_df.empty:
                daily_totals_sorted_for_rolling = daily_totals_df.sort_values(by='DateOnly')
                if len(daily_totals_sorted_for_rolling) >= 1:
                    avg_daily_volume_rolling_7_days = daily_totals_sorted_for_rolling['BlockVolume'].rolling(window=7, min_periods=1).mean().iloc[-1]
                else: 
                     avg_daily_volume_rolling_7_days = 0

            print("Historical trend calculation completed.")
            return {
                "reference_date_for_trends": self.target_date.strftime("%Y-%m-%d"),
                "trend_period_days": days_prior,
                "num_weeks_for_avg": num_weeks_for_avg,
                "short_term_ma_occurrences": short_term_ma_occurrences,
                "three_hour_block_trends": block_trends,
                "daily_summary_trends": dict(daily_summary),
                "overall_summary": {
                    f"avg_daily_volume_last_{days_prior}_days": round(overall_avg_daily_45_days, 2),
                    "avg_daily_volume_rolling_7_days": round(avg_daily_volume_rolling_7_days, 2) 
                }
            }

        except Exception as e:
            print(f"Error in calculate_historical_trends: {str(e)}")
            traceback.print_exc()
            return {
                "reference_date_for_trends": self.target_date.strftime("%Y-%m-%d") if hasattr(self, 'target_date') else "N/A",
                "trend_period_days": days_prior,
                "error_message": str(e)
            }
    
    def log_scaling_metrics(self, scaling_data):
        """Log scaling metrics for analysis"""
        try:
            metrics = scaling_data['metrics']
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = {
                'timestamp': timestamp,
                'weighted_trend': metrics.get('weighted_trend', 0),
                'avg_scaling': metrics.get('avg_scaling', 0),
                'base_scaling': scaling_data['base_scaling'],
                'next_day_scaling': scaling_data['next_day_scaling']
            }
            
            # Log to CloudWatch or local file
            print("\nScaling Metrics Log:")
            for key, value in log_entry.items():
                print(f"{key}: {value}")
                
        except Exception as e:
            print(f"Error logging scaling metrics: {str(e)}")
    
    def get_previous_year_data(self, date):
        """Get historical data from previous year"""
        try:
            prev_year_data = self.df[
                (self.df['Time'].dt.month == date.month) & 
                (self.df['Time'].dt.day == date.day) &
                (self.df['Time'].dt.year == date.year)
            ]
            
            if not prev_year_data.empty:
                return [{
                    "Time": row['Time'].strftime('%Y-%m-%dT%H:00'),
                    "Workable": float(row['Workable'])
                } for _, row in prev_year_data.iterrows()]
            return []
        except Exception as e:
            print(f"Error getting previous year data: {str(e)}")
            return []
    
    def get_current_day_data(self, current_time):
        """Get actual data for current day"""
        try:
            current_date = current_time.date()
            current_day_data = self.df[
                self.df['Time'].dt.date == current_date
            ]
            
            if not current_day_data.empty:
                return [{
                    "Time": row['Time'].strftime('%Y-%m-%dT%H:00'),
                    "Workable": float(row['Workable'])
                } for _, row in current_day_data.iterrows()]
            return []
        except Exception as e:
            print(f"Error getting current day data: {str(e)}")
            return []
    
    def predict_next_day_enhanced(self, target_date):
        """Enhanced next-day prediction incorporating network scaling factors and zeroing midnight"""
        try:
            # Get network scaling factors
            scaling_data = self.get_network_scaling_factors()
            if not scaling_data:
                return None
    
            # Create prediction range
            pred_range = pd.date_range(
                start=target_date.replace(hour=0),
                periods=24,
                freq='h'
            )
    
            # Get base SARIMA predictions
            base_predictions = self.sarima_results.predict(
                start=pred_range[0],
                end=pred_range[-1]
            )
    
            # Apply network-based scaling
            adjusted_predictions = []
            for hour, base_pred in enumerate(base_predictions):
                hour_factor = scaling_data['hourly_factors'][hour]
                adjusted_pred = base_pred * hour_factor
                adjusted_predictions.append(max(0, round(adjusted_pred)))
    
            # Get the midnight prediction value
            midnight_value = adjusted_predictions[0]
    
            # Adjust predictions based on midnight value
            final_predictions = []
            for hour, pred in enumerate(adjusted_predictions):
                if hour == 0:
                    final_predictions.append(0)
                else:
                    adjusted_value = max(0, pred - midnight_value)
                    final_predictions.append(adjusted_value)
    
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Time': pred_range.strftime('%Y-%m-%dT%H:00'),
                'Predicted_Workable': final_predictions
            })
    
            print("\nNetwork Prediction Metrics:")
            print(f"Current Day Target (IPT): {scaling_data['metrics']['current_ipt']:,.0f}")
            print(f"3-Day Average Target: {scaling_data['metrics']['three_day_ipt']:,.0f}")
            print(f"Trend Factor: {scaling_data['metrics']['trend_factor']:.2f}")
            print(f"Base Scaling: {scaling_data['base_scaling']:.2f}")
            print(f"Midnight adjustment applied: -{midnight_value}")
    
            return results_df
    
        except Exception as e:
            print(f"Error in enhanced next-day prediction: {str(e)}")
            traceback.print_exc()
            return None
    

def check_aws_credentials():
    try:
        # Try to get the caller identity
        sts = boto3.client('sts')
        account = sts.get_caller_identity()
        print("AWS Credentials are configured correctly!")
        print(f"Account ID: {account['Account']}")
        print(f"User/Role ARN: {account['Arn']}")
        return True
    except Exception as e:
        print("AWS Credentials check failed:", str(e))
        return False

def setup_s3_bucket():
    try:
        # Create S3 client with specific region
        s3 = boto3.client('s3', region_name='us-west-1')
        bucket_name = 'ledger-prediction-charting-008971633421'
        
        print(f"Attempting to create/access bucket: {bucket_name}")
        
        # Check if bucket exists
        try:
            s3.head_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} already exists")
        except:
            # Create bucket with location constraint for us-west-1
            print(f"Creating bucket {bucket_name}...")
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={
                    'LocationConstraint': 'us-west-1'
                }
            )
            
            print("Bucket created, setting up configurations...")
            
            # Enable versioning
            s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            print(f"Bucket {bucket_name} created and configured successfully!")
        
        # Store the bucket name for use in other functions
        global S3_BUCKET_NAME
        S3_BUCKET_NAME = bucket_name
        
        return True
    except Exception as e:
        print(f"Error setting up S3 bucket: {str(e)}")
        print(f"Full error details: {str(type(e).__name__)}: {str(e)}")
        return False


def check_aws_region():
    try:
        session = boto3.Session()
        current_region = session.region_name
        print(f"Current AWS region: {current_region}")
        return current_region
    except Exception as e:
        print(f"Error checking AWS region: {str(e)}")
        return None


def upload_to_s3(local_file, bucket, s3_file):
    """Upload a file to an S3 bucket"""
    s3_client = boto3.client('s3', region_name='us-west-1')
    try:
        print(f"Attempting to upload {local_file} to {bucket}/{s3_file}")
        
        # Upload the file
        s3_client.upload_file(local_file, bucket, s3_file)
        
        # Get the URL (note: this will only work if you configure bucket access later)
        url = f"https://{bucket}.s3.us-west-1.amazonaws.com/{s3_file}"
        print(f"File uploaded successfully to: {url}")
        return True
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(f"An error occurred during upload: {str(e)}")
        return False

def main():
    check_environment()
    try:
        if not check_aws_credentials():
            raise Exception("AWS credentials not properly configured")
        region = check_aws_region()
        if not region:
            raise Exception("AWS region not properly configured")
        if not setup_s3_bucket():
            raise Exception("Failed to setup S3 bucket")
        
        predictor = ChargePredictor()
        
        print("\nGenerating SARIMA predictions for current day...")
        sarima_results = predictor.predict_for_target_date()
        
        print("\nGenerating RF predictions for current day...")
        rf_results = predictor.get_rf_predictions()
        
        print("\nGenerating 48-hour rolling predictions...")
        rolling_predictions = predictor.predict_rolling_48h()
        
        next_day_target = predictor.target_date + timedelta(days=1)
        print(f"\nGenerating predictions for next day: {next_day_target.date()}")
        next_day_sarima_enhanced = predictor.predict_next_day_enhanced(next_day_target)
        
        print("\nGetting ledger information...")
        ledger_info = predictor.get_ledger_information()
        
        current_date_dt = predictor.target_date # This is already a datetime object from pd.to_datetime('today').normalize()
        prev_year_date = current_date_dt - pd.DateOffset(years=1)
        next_day_prev_year_date = next_day_target - pd.DateOffset(years=1)

        prev_year_records = predictor.get_previous_year_data(prev_year_date)
        next_day_prev_year_records = predictor.get_previous_year_data(next_day_prev_year_date)
        current_day_records = predictor.get_current_day_data(current_time=datetime.combine(current_date_dt, datetime.min.time()))


        no_same_day_current = predictor.predict_without_same_day_influence(predictor.target_date)

        historical_trends_data = predictor.calculate_historical_trends(days_prior=45, num_weeks_for_avg=6, short_term_ma_occurrences=3)

        json_data = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_day": {
                "date": predictor.target_date.strftime("%Y-%m-%d"),
                "network_prediction": predictor.network_prediction,
                "sarima_predictions": sarima_results.to_dict(orient='records') if sarima_results is not None else [],
                "rf_predictions": rf_results.to_dict(orient='records') if rf_results is not None else [],
                "predictions_no_same_day": no_same_day_current.to_dict(orient='records') if no_same_day_current is not None else [],
                "previous_year_data": prev_year_records,
                "current_day_data": current_day_records
            },
            "next_day": {
                "date": next_day_target.strftime("%Y-%m-%d"),
                "sarima_predictions": next_day_sarima_enhanced.to_dict(orient='records') if next_day_sarima_enhanced is not None else [],
                "rf_predictions": rf_results.to_dict(orient='records') if rf_results is not None else [], 
                "previous_year_data": next_day_prev_year_records
            },
            "extended_predictions": {
                "predictions": rolling_predictions.to_dict(orient='records') if rolling_predictions is not None else []
            },
            "Ledger_Information": ledger_info if ledger_info else {},
            "historical_context": historical_trends_data 
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_file = os.path.join(current_dir, 'VIZ.json')
        
        print(f"\nWriting to JSON file at: {local_file}")
        
        try:
            with open(local_file, 'w') as f:
                json.dump(json_data, f, indent=4)
            print("Successfully wrote to VIZ.json")
            
            if S3_BUCKET_NAME: 
                s3_file_name = f"predictions/{datetime.now().strftime('%Y-%m-%d_%H')}/VIZ.json"
                if upload_to_s3(local_file, S3_BUCKET_NAME, s3_file_name):
                    print(f"Successfully uploaded {local_file} to S3 bucket {S3_BUCKET_NAME}")
                else:
                    print(f"Failed to upload {local_file} to S3 bucket {S3_BUCKET_NAME}")
            else:
                print("No valid S3 bucket name available for upload (S3_BUCKET_NAME not set).")
                
        except Exception as e:
            print(f"Error writing/uploading JSON file: {str(e)}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"\nError in main: {str(e)}")
        traceback.print_exc()


#'''

if __name__ == "__main__":
    main()


'''
def signal_handler(sig, frame):
    print(f"\nShutdown initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Gracefully shutting down... (This might take a moment)")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    start_time = datetime.now()
    print(f"\nScheduler started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Setting up daily runs at 08:30, 20:40, and 23:15")
    print("\nPress Ctrl+C to exit gracefully")
    
    # Calculate and display time until next run
    now = datetime.now()
    scheduled_times = ["08:30", "12:15", "20:30", "23:15"]  # Made consistent with scheduled tasks
    next_run = None
    
    for time_str in scheduled_times:
        hours, minutes = map(int, time_str.split(':'))
        potential_next = now.replace(hour=hours, minute=minutes, second=0, microsecond=0)
        
        if potential_next <= now:
            potential_next += timedelta(days=1)
            
        if next_run is None or potential_next < next_run:
            next_run = potential_next
    
    wait_time = (next_run - now).total_seconds() / 60
    
    print(f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Time until next run: {wait_time:.1f} minutes")
    
    # Schedule the tasks
    schedule.every().day.at("08:30").do(run_scheduled_task)
    schedule.every().day.at("12:15").do(run_scheduled_task)  # Changed from 20:35 to 20:45    schedule.every().day.at("08:30").do(run_scheduled_task)
    schedule.every().day.at("20:30").do(run_scheduled_task)
    schedule.every().day.at("23:15").do(run_scheduled_task)
    
    run_count = 0
    
    # Run forever until Ctrl+C
    try:
        while True:
            schedule.run_pending()
            
            # Update run count and display status every 30 seconds
            if datetime.now().minute % 10 == 0 and datetime.now().second == 0:
                current_time = datetime.now()
                uptime = current_time - start_time
                next_job = schedule.next_run()
                time_to_next = (next_job - current_time).total_seconds() / 60 if next_job else 0
                
                print(f"\nStatus Update ({current_time.strftime('%Y-%m-%d %H:%M:%S')}):")
                print(f"Uptime: {uptime}")
                print(f"Runs completed: {run_count}")
                print(f"Next run in: {time_to_next:.1f} minutes")
                print("Press Ctrl+C to exit")
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\nShutdown initiated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {datetime.now() - start_time}")
        print(f"Total runs completed: {run_count}")
        print("Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError in scheduler: {str(e)}")
        traceback.print_exc()
        print("\nPress Ctrl+C to exit")
#'''

