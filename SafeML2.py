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
import signal
import json
import boto3
import io
import schedule
import time
from datetime import datetime
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
        """Train the SARIMA model"""
        try:
            # Prepare time series data
            self.ts_data = self.df.set_index('Time')['Workable']
            self.ts_data = self.ts_data.resample('h').mean()
            self.ts_data = self.ts_data.ffill()
            
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

    def get_network_scaling_factors(self):
        """Calculate scaling factors based on network predictions with stronger IPT3DAY influence"""
        try:
            # Get network predictions from granular.csv
            granular_obj = self.s3.get_object(
                Bucket=self.bucket_name,
                Key='userloads/granular.csv'
            )
            granular_data = granular_obj['Body'].read().decode('utf-8')
            granular_df = pd.read_csv(io.StringIO(granular_data))

            # Get the most recent values
            current_target = float(granular_df['IPTNW'].iloc[0])  # Current day target
            three_day_target = float(granular_df['IPT3DAY'].iloc[0])  # 3-day average

            # Calculate trend factor with stronger weight towards IPT3DAY
            trend_factor = three_day_target / current_target

            # Calculate dynamic scaling factors with stronger adjustment
            if current_target < three_day_target:
                # Upward trend expected - inflate predictions
                base_scaling = 1 + (trend_factor - 1) * 0.85  # Increased from 0.7 to 0.85
                next_day_scaling = min(1.6, base_scaling)  # Increased cap to 60% increase
            else:
                # Downward trend expected - deflate predictions
                base_scaling = 1 - (1 - trend_factor) * 0.85  # Increased from 0.7 to 0.85
                next_day_scaling = max(0.65, base_scaling)  # Increased floor drop to 35%

            # Adjusted time-of-day factors to align more closely with IPT3DAY
            hourly_factors = {}
            for hour in range(24):
                if hour < 6:  # Early morning
                    hourly_factors[hour] = next_day_scaling * 0.90  # Increased from 0.85
                elif 6 <= hour < 12:  # Morning peak
                    hourly_factors[hour] = next_day_scaling * 1.10  # Slightly reduced from 1.15
                elif 12 <= hour < 18:  # Afternoon
                    hourly_factors[hour] = next_day_scaling * 1.07  # Increased from 1.0
                else:  # Evening
                    hourly_factors[hour] = next_day_scaling * 1.05  # Increased from 0.90

            # Calculate average scaling factor for validation
            avg_scaling = sum(hourly_factors.values()) / 24
            target_ratio = three_day_target / current_target

            print("\nScaling Factor Analysis:")
            print(f"Target Ratio (IPT3DAY/IPTNW): {target_ratio:.3f}")
            print(f"Average Scaling Factor: {avg_scaling:.3f}")
            print(f"Base Scaling: {base_scaling:.3f}")
            print(f"Next Day Scaling: {next_day_scaling:.3f}")

            return {
                'base_scaling': base_scaling,
                'next_day_scaling': next_day_scaling,
                'hourly_factors': hourly_factors,
                'metrics': {
                    'current_target': current_target,
                    'three_day_target': three_day_target,
                    'trend_factor': trend_factor,
                    'target_ratio': target_ratio,
                    'avg_scaling': avg_scaling
                }
            }

        except Exception as e:
            print(f"Error calculating network scaling factors: {str(e)}")
            traceback.print_exc()
            return None


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

            # Get base SARIMA
            base_predictions = self.sarima_results.predict(
                start=pred_range[0],
                end=pred_range[-1]
            )

            # Apply network-based scaling
            adjusted_predictions = []
            for hour, base_pred in enumerate(base_predictions):
                hour_factor = scaling_data['hourly_factors'][hour]
                
                # Apply scaling
                adjusted_pred = base_pred * hour_factor
                
                # Ensure non-negative and round
                adjusted_predictions.append(max(0, round(adjusted_pred)))

            # Get the midnight prediction value (hour 00:00)
            midnight_value = adjusted_predictions[0]
            print(f"\nOriginal midnight prediction: {midnight_value}")

            # Adjust all predictions based on midnight difference
            final_predictions = []
            for hour, pred in enumerate(adjusted_predictions):
                if hour == 0:
                    # Set midnight to 0
                    final_predictions.append(0)
                else:
                    # Adjust other hours by subtracting the midnight value
                    adjusted_value = max(0, pred - midnight_value)
                    final_predictions.append(adjusted_value)

            # Create simplified results DataFrame
            results_df = pd.DataFrame({
                'Time': pred_range.strftime('%Y-%m-%dT%H:00'),
                'Predicted_Workable': final_predictions
            })

            print("\nNetwork Prediction Metrics:")
            print(f"Current Day Target (IPTNW): {scaling_data['metrics']['current_target']:,.0f}")
            print(f"3-Day Average Target (IPT3DAY): {scaling_data['metrics']['three_day_target']:,.0f}")
            print(f"Trend Factor: {scaling_data['metrics']['trend_factor']:.2f}")
            print(f"Base Scaling: {scaling_data['base_scaling']:.2f}")
            print(f"Midnight adjustment applied: -{midnight_value}")

            return results_df

        except Exception as e:
            print(f"Error in enhanced next-day prediction: {str(e)}")
            traceback.print_exc()
            return None




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
            
        # Check AWS region
        region = check_aws_region()
        if not region:
            raise Exception("AWS region not properly configured")
            
        # Setup S3 bucket
        if not setup_s3_bucket():
            raise Exception("Failed to setup S3 bucket")
        
        predictor = ChargePredictor()
        
        # Generate SARIMA predictions for current day
        print("\nGenerating SARIMA predictions for current day...")
        sarima_results = predictor.predict_for_target_date()
        print(f"SARIMA predictions shape: {sarima_results.shape}")
        
        # Generate RF predictions for current day
        print("\nGenerating RF predictions for current day...")
        rf_results = predictor.get_rf_predictions()
        print(f"RF predictions shape: {rf_results.shape}")
        
        # Write both results to Excel
        print("\nWriting to Excel...")
        # predictor.write_predictions_to_excel(sarima_results, rf_results)
        
        next_day_target = predictor.target_date + timedelta(days=1)
        print(f"\nGenerating predictions for next day: {next_day_target.date()}")
        
        print("\nGetting ledger information...")
        ledger_info = predictor.get_ledger_information()
        
        # Temporarily store original target date
        original_target = predictor.target_date
        
        # Set target date to next day
        predictor.target_date = next_day_target
        
        # Generate both original and enhanced predictions for next day
        next_day_sarima = predictor.predict_for_target_date()
        next_day_rf = predictor.get_rf_predictions()
        next_day_enhanced = predictor.predict_next_day_enhanced(next_day_target)  # New enhanced predictions
        
        # Restore original target date
        predictor.target_date = original_target
        
        # Get previous year's date
        current_date = predictor.target_date
        prev_year_date = current_date - pd.DateOffset(years=1)
        next_day_prev_year_date = next_day_target - pd.DateOffset(years=1)

        # Get previous year's data
        prev_year_data = predictor.df[
            (predictor.df['Time'].dt.month == prev_year_date.month) & 
            (predictor.df['Time'].dt.day == prev_year_date.day) &
            (predictor.df['Time'].dt.year == prev_year_date.year)
        ]
        next_day_prev_year_data = predictor.df[
            (predictor.df['Time'].dt.month == next_day_prev_year_date.month) & 
            (predictor.df['Time'].dt.day == next_day_prev_year_date.day) &
            (predictor.df['Time'].dt.year == next_day_prev_year_date.year)
        ]
        
        # Format previous year's data for JSON
        if not prev_year_data.empty:
            prev_year_records = []
            for _, row in prev_year_data.iterrows():
                prev_year_records.append({
                    "Time": row['Time'].strftime('%Y-%m-%dT%H:00'),
                    "Workable": float(row['Workable'])
                })
        else:
            prev_year_records = []

        if not next_day_prev_year_data.empty:
            next_day_prev_year_records = []
            for _, row in next_day_prev_year_data.iterrows():
                next_day_prev_year_records.append({
                    "Time": row['Time'].strftime('%Y-%m-%dT%H:00'),
                    "Workable": float(row['Workable'])
                })
        else:
            next_day_prev_year_records = []

        current_date = predictor.target_date.date()
        current_day_data = predictor.df[
            predictor.df['Time'].dt.date == current_date
        ]
        
        # Format current day's data
        current_day_records = []
        if not current_day_data.empty:
            for _, row in current_day_data.iterrows():
                current_day_records.append({
                    "Time": row['Time'].strftime('%Y-%m-%dT%H:00'),
                    "Workable": float(row['Workable'])
                })
        no_same_day_current = predictor.predict_without_same_day_influence(predictor.target_date)
        #no_same_day_next = predictor.predict_without_same_day_influence(next_day_target)

        # Create JSON with results
        json_data = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current_day": {
                "date": predictor.target_date.strftime("%Y-%m-%d"),
                "network_prediction": predictor.network_prediction,
                "sarima_predictions": sarima_results.to_dict(orient='records'),
                "rf_predictions": rf_results.to_dict(orient='records'),
                "predictions_no_same_day": no_same_day_current.to_dict(orient='records'),
                "previous_year_data": prev_year_records,
                "current_day_data": current_day_records
            },
            "next_day": {
                "date": next_day_target.strftime("%Y-%m-%d"),
                "sarima_predictions": next_day_enhanced.to_dict(orient='records'),
                "rf_predictions": next_day_rf.to_dict(orient='records'),
                "previous_year_data": next_day_prev_year_records
            },
            "Ledger_Information": ledger_info if ledger_info else {}
        }
        # Write to JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_file = os.path.join(current_dir, 'VIZ.json')
        
        print(f"Writing to JSON file at: {local_file}")
        
        try:
            # Write to JSON file (will create if doesn't exist, overwrite if it does)
            with open(local_file, 'w') as f:
                json.dump(json_data, f, indent=4)
            print("Successfully wrote to VIZ.json")
            
            # Upload to S3 if bucket name is available
            if S3_BUCKET_NAME:
                s3_file_name = f"predictions/{datetime.now().strftime('%Y-%m-%d_%H')}/VIZ.json"
                
                if upload_to_s3(local_file, S3_BUCKET_NAME, s3_file_name):
                    print(f"Successfully uploaded {local_file} to S3 bucket {S3_BUCKET_NAME}")
                else:
                    print(f"Failed to upload {local_file} to S3 bucket {S3_BUCKET_NAME}")
            else:
                print("No valid S3 bucket name available for upload")
                
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
    print("Setting up hourly runs at :15 past each hour")
    print("\nPress Ctrl+C to exit gracefully")
    
    # Calculate and display time until next run
    now = datetime.now()
    next_run = now.replace(minute=15, second=0, microsecond=0)
    if now.minute >= 15:
        next_run = next_run + timedelta(hours=1)
    wait_time = (next_run - now).total_seconds() / 60
    
    print(f"Next run scheduled for: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Time until next run: {wait_time:.1f} minutes")
    
    # Schedule the task to run every hour at :15
    schedule.every().hour.at(":15").do(run_scheduled_task)
    
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

'''