import boto3
import json
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse

class SARIMAAnalyzer:
    def __init__(self, reanalyze_all=False):
        self.s3 = boto3.client('s3')
        self.bucket_name = 'ledger-prediction-charting-008971633421'
        self.stats_file = 'sarima_stats.json'
        self.reanalyze_all = reanalyze_all
        self.stats = self.load_stats()

    def load_stats(self):
        if os.path.exists(self.stats_file) and not self.reanalyze_all:
            with open(self.stats_file, 'r') as f:
                return json.load(f)
        return {}

    def save_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def get_available_prediction_days(self):
        """Get list of days that have hour 23 predictions available"""
        available_days = set()
        try:
            # List objects in predictions folder
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix='predictions/'):
                for obj in page.get('Contents', []):
                    # Look for folders with _23
                    if '_23/VIZ.json' in obj['Key']:
                        # Extract date from key
                        date_str = obj['Key'].split('/')[1][:10]  # Get YYYY-MM-DD part
                        available_days.add(date_str)
            
            print(f"Found {len(available_days)} days with predictions available")
            return sorted(list(available_days))
        except Exception as e:
            print(f"Error getting available prediction days: {e}")
            return []

    def get_prediction_data(self, date):
        """Get prediction data for a specific date"""
        try:
            key = f'predictions/{date}_23/VIZ.json'
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            return data['current_day']['predictions_no_same_day']
        except Exception as e:
            print(f"Error fetching prediction data for {date}: {e}")
            return None

    def get_actual_data(self):
        """Get actual data from combine.csv"""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key='userloads/combine.csv')
            data = response['Body'].read().decode('utf-8').splitlines()
            
            # Parse the CSV data
            records = []
            for line in data:
                if line.strip():  # Skip empty lines
                    try:
                        timestamp, workable = line.split(',')
                        if timestamp.lower() != 'time':  # Skip header if present
                            records.append({
                                'Time': pd.to_datetime(timestamp),
                                'Workable': float(workable)
                            })
                    except (ValueError, pd.errors.ParserError) as e:
                        print(f"Skipping invalid line: {line} - Error: {e}")
                        continue

            df = pd.DataFrame(records)
            if not df.empty:
                print(f"Loaded {len(df)} actual data points")
                print(f"Date range: {df['Time'].min()} to {df['Time'].max()}")
                return df
            else:
                print("No valid data points found in combine.csv")
                return None
        except Exception as e:
            print(f"Error fetching actual data: {e}")
            print(f"First few lines of data:")
            try:
                print('\n'.join(data[:5]))
            except:
                pass
            return None

    def analyze_day(self, date, predictions, actuals):
        """Analyze predictions vs actuals for a single day with enhanced metrics"""
        day_stats = {
            'date': date,
            'total_predicted': 0,
            'total_actual': 0,
            'hourly_errors': [],
            'mape': 0,
            'over_predictions': 0,
            'under_predictions': 0,
            'hourly_stats': {},  # New: Track each hour's performance
            'percent_differences': [],  # New: Track percentage differences
            'end_of_day_accuracy': {},  # New: Specific end-of-day metrics
            'peak_hours_accuracy': {},  # New: Track accuracy during peak hours (e.g., 9-17)
            'worst_performing_hours': []  # New: Track the worst performing hours
        }

        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Analyze each hour
        for pred in predictions:
            pred_time = datetime.strptime(pred['Time'], '%Y-%m-%dT%H:00')
            if pred_time.hour == 0:  # Skip midnight
                continue
            actual_data = actuals[actuals['Time'] == pred_time]
            
            if not actual_data.empty:
                hour = pred_time.hour
                pred_value = pred['Predicted_Workable_No_Same_Day']
                actual_value = actual_data['Workable'].iloc[0]
                
                day_stats['total_predicted'] += pred_value
                day_stats['total_actual'] += actual_value
                
                # Hourly analysis
                hour_stats = {
                    'predicted': pred_value,
                    'actual': actual_value,
                    'difference': pred_value - actual_value,
                    'percent_difference': ((pred_value - actual_value) / actual_value * 100) if actual_value != 0 else 0
                }
                
                day_stats['hourly_stats'][hour] = hour_stats
                
                if actual_value != 0:
                    error = abs((pred_value - actual_value) / actual_value)
                    day_stats['hourly_errors'].append(error)
                    day_stats['percent_differences'].append(hour_stats['percent_difference'])
                
                if pred_value > actual_value:
                    day_stats['over_predictions'] += 1
                elif pred_value < actual_value:
                    day_stats['under_predictions'] += 1

        # Calculate MAPE
        if day_stats['hourly_errors']:
            day_stats['mape'] = sum(day_stats['hourly_errors']) / len(day_stats['hourly_errors']) * 100

        # End of day accuracy (last 3 hours)
        end_hours = [21, 22, 23]
        end_stats = {hour: day_stats['hourly_stats'].get(hour, {}) for hour in end_hours}
        day_stats['end_of_day_accuracy'] = {
            'average_error': sum(abs(stats.get('percent_difference', 0)) for stats in end_stats.values()) / len(end_stats) if end_stats else 0,
            'final_hour_error': abs(day_stats['hourly_stats'].get(23, {}).get('percent_difference', 0))
        }

        # Peak hours accuracy (9-17)
        peak_hours = range(9, 18)
        peak_stats = {hour: day_stats['hourly_stats'].get(hour, {}) for hour in peak_hours}
        day_stats['peak_hours_accuracy'] = {
            'average_error': sum(abs(stats.get('percent_difference', 0)) for stats in peak_stats.values()) / len(peak_stats) if peak_stats else 0
        }

        # Find worst performing hours (highest absolute percentage difference)
        hourly_errors = [(hour, abs(stats.get('percent_difference', 0))) 
                        for hour, stats in day_stats['hourly_stats'].items()]
        worst_hours = sorted(hourly_errors, key=lambda x: x[1], reverse=True)[:3]
        day_stats['worst_performing_hours'] = worst_hours

        return day_stats

    def run_analysis(self):
        """Run the analysis for all available days"""
        print("Starting SARIMA prediction analysis...")
        
        available_days = self.get_available_prediction_days()
        if not available_days:
            print("No prediction days found")
            return

        actuals = self.get_actual_data()
        if actuals is None:
            return

        new_days_analyzed = 0
        for date in available_days:
            if self.reanalyze_all or date not in self.stats:
                predictions = self.get_prediction_data(date)
                if predictions:
                    day_stats = self.analyze_day(date, predictions, actuals)
                    self.stats[date] = day_stats
                    new_days_analyzed += 1
                    print(f"Analyzed {date}")

        if new_days_analyzed > 0:
            print(f"\nAnalyzed {new_days_analyzed} days")
            self.save_stats()
        else:
            print("\nNo new days to analyze")

        self.print_summary()

    def print_summary(self):
        """Print enhanced analysis summary and recommendations"""
        if not self.stats:
            print("No data to summarize")
            return

        print("\nSARIMA Prediction Analysis Summary")
        print("=" * 50)
        
        days = len(self.stats)
        
        # Initialize aggregated metrics
        total_mape = 0
        total_over = 0
        total_under = 0
        hourly_errors = {hour: [] for hour in range(1, 24)}
        end_day_errors = []
        peak_hours_errors = []
        all_percent_differences = []
        
        # Collect metrics across all days
        for date, stats in self.stats.items():
            total_mape += stats.get('mape', 0)
            total_over += stats.get('over_predictions', 0)
            total_under += stats.get('under_predictions', 0)
            
            # Collect hourly errors
            for hour, hour_stats in stats.get('hourly_stats', {}).items():
                hourly_errors[hour].append(abs(hour_stats.get('percent_difference', 0)))
                all_percent_differences.append(hour_stats.get('percent_difference', 0))
            
            # Collect end of day accuracy
            end_day_errors.append(stats.get('end_of_day_accuracy', {}).get('final_hour_error', 0))
            
            # Collect peak hours accuracy
            peak_hours_errors.append(stats.get('peak_hours_accuracy', {}).get('average_error', 0))

        # Calculate averages
        avg_mape = total_mape / days if days > 0 else 0
        avg_over = total_over / days if days > 0 else 0
        avg_under = total_under / days if days > 0 else 0
        
        # Find worst performing hours overall
        avg_hourly_errors = {hour: sum(errors)/len(errors) if errors else 0 
                            for hour, errors in hourly_errors.items()}
        worst_hours = sorted(avg_hourly_errors.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"\nOverall Statistics ({days} days analyzed)")
        print("-" * 40)
        print(f"Average MAPE: {avg_mape:.2f}%")
        print(f"Average over-predictions per day: {avg_over:.2f}")
        print(f"Average under-predictions per day: {avg_under:.2f}")
        if all_percent_differences:
            print(f"Average percentage difference: {sum(all_percent_differences)/len(all_percent_differences):.2f}%")
        if end_day_errors:
            print(f"End of day accuracy (Hour 23): {sum(end_day_errors)/len(end_day_errors):.2f}%")
        if peak_hours_errors:
            print(f"Peak hours accuracy (9-17): {sum(peak_hours_errors)/len(peak_hours_errors):.2f}%")

        print("\nWorst Performing Hours")
        print("-" * 40)
        for hour, error in worst_hours:
            print(f"Hour {hour:02d}:00 - Average error: {error:.2f}%")

        print("\nRecent Performance Trend (Last 5 days)")
        print("-" * 40)
        recent_dates = sorted(self.stats.keys())[-5:]
        for date in recent_dates:
            stats = self.stats[date]
            print(f"{date}:")
            print(f"  MAPE: {stats.get('mape', 0):.2f}%")
            print(f"  End of day error: {stats.get('end_of_day_accuracy', {}).get('final_hour_error', 0):.2f}%")
            print(f"  Peak hours error: {stats.get('peak_hours_accuracy', {}).get('average_error', 0):.2f}%")

        # Enhanced recommendations
        print("\nDetailed Recommendations")
        print("-" * 40)
        if avg_mape > 15:
            print("- High overall error rate detected:")
            print(f"  * Consider retraining SARIMA model")
            print(f"  * Focus on worst performing hours: {', '.join(f'{hour:02d}:00' for hour, _ in worst_hours[:3])}")
        
        if sum(end_day_errors)/len(end_day_errors) > 15:
            print("- End of day predictions need improvement:")
            print("  * Consider adjusting evening hour parameters")
            print("  * Review factors affecting end-of-day workload")

        if sum(peak_hours_errors)/len(peak_hours_errors) > 15:
            print("- Peak hours accuracy needs attention:")
            print("  * Review business hour prediction parameters")
            print("  * Consider adding peak hour specific adjustments")
        
        print("\nPattern Analysis")
        print("-" * 40)
        print(f"Morning accuracy (0-6): {self._calculate_period_accuracy(hourly_errors, range(1,7)):.2f}%")
        print(f"Business hours accuracy (6-18): {self._calculate_period_accuracy(hourly_errors, range(7,19)):.2f}%")
        print(f"Evening accuracy (18-23): {self._calculate_period_accuracy(hourly_errors, range(19,24)):.2f}%")
        
    def _calculate_period_accuracy(self, hourly_errors, hours):
        """Helper method to calculate accuracy for a specific period"""
        period_errors = []
        for hour in hours:
            if hourly_errors[hour]:
                period_errors.extend(hourly_errors[hour])
        return sum(period_errors)/len(period_errors) if period_errors else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARIMA Prediction Analyzer")
    parser.add_argument('--reanalyze', action='store_true', help="Reanalyze all available days")
    args = parser.parse_args()

    analyzer = SARIMAAnalyzer(reanalyze_all=args.reanalyze)
    analyzer.run_analysis()