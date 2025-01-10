import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from holidays import US as us_holidays  # You'll need to pip install holidays
import tkinter as tk
from tkinter import ttk
from tkcalendar import Calendar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk


class ChargePredictor:
    def __init__(self, csv_path):
        print(f"Loading data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.df['Time'] = pd.to_datetime(self.df['Time'])
        
        print("\nDataset Information:")
        print(f"Total rows: {len(self.df)}")
        print(f"Date range: {self.df['Time'].min()} to {self.df['Time'].max()}")
        print(f"Number of unique dates: {self.df['Time'].dt.date.nunique()}")
        print("\nSample of data:")
        print(self.df[['Time', 'Workable']].head())
        
        # Verify we have the required columns
        required_columns = ['Time', 'Workable']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        self.prepare_data()
        self.train_ml_model()
    
    

    def prepare_data(self):
        """Prepare and clean the data"""
        # Existing time features
        self.df['Hour'] = self.df['Time'].dt.hour
        self.df['DayOfWeek'] = self.df['Time'].dt.day_name()
        self.df['Month'] = self.df['Time'].dt.month
        self.df['DayOfMonth'] = self.df['Time'].dt.day
        self.df['Date'] = self.df['Time'].dt.date
        
        # New features
        self.df['Year'] = self.df['Time'].dt.year
        self.df['WeekOfYear'] = self.df['Time'].dt.isocalendar().week
        self.df['IsWeekend'] = self.df['Time'].dt.weekday >= 5
        
        # Holiday features
        self.df['IsHoliday'] = self.df['Time'].dt.date.apply(lambda x: x in us_holidays())
        self.df['IsNearHoliday'] = self.df.apply(self._is_near_holiday, axis=1)
        
        # Season features
        self.df['Season'] = self.df['Month'].apply(self._get_season)
        
        # Peak time features
        self.df['IsPeakHour'] = self.df['Hour'].apply(
            lambda x: 1 if (9 <= x <= 17) or (6 <= x <= 8) else 0
        )

        # Clean data
        print("\nData cleaning summary:")
        print(f"Total rows before cleaning: {len(self.df)}")
        print(f"NaN values in Workable column: {self.df['Workable'].isna().sum()}")
        
        # Handle missing values
        # Option 1: Remove rows with NaN values
        self.df = self.df.dropna(subset=['Workable'])
        
        # Option 2 (alternative): Fill NaN values with mean for that hour
        # self.df['Workable'] = self.df.groupby('Hour')['Workable'].transform(
        #     lambda x: x.fillna(x.mean())
        # )
        
        print(f"Total rows after cleaning: {len(self.df)}")
        print(f"NaN values remaining: {self.df['Workable'].isna().sum()}")

    def _is_near_holiday(self, row):
        """Check if date is within 3 days of a holiday"""
        date = row['Time'].date()
        for holiday_date in us_holidays().keys():
            if abs((date - holiday_date).days) <= 3:
                return True
        return False

    def _get_season(self, month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    def get_historical_data(self, target_date):
        """Get data from the previous two years"""
        try:
            target_date = pd.to_datetime(target_date)
            one_year_ago = target_date - pd.DateOffset(years=1)
            two_years_ago = target_date - pd.DateOffset(years=2)
            
            print(f"\nSearching for historical data:")
            print(f"Target date: {target_date.date()}")
            print(f"Looking for one year ago: {one_year_ago.date()}")
            print(f"Looking for two years ago: {two_years_ago.date()}")
            
            # Get data for both previous years
            historical_data = self.df[
                ((self.df['Time'].dt.month == one_year_ago.month) & 
                 (self.df['Time'].dt.day == one_year_ago.day) &
                 (self.df['Time'].dt.year == one_year_ago.year)) |
                ((self.df['Time'].dt.month == two_years_ago.month) & 
                 (self.df['Time'].dt.day == two_years_ago.day) &
                 (self.df['Time'].dt.year == two_years_ago.year))
            ].copy()
            
            if historical_data.empty:
                print("No historical data found for exact dates")
                # Look for nearby dates in both years
                nearby_dates = self.df[
                    ((self.df['Time'].dt.month == one_year_ago.month) &
                     (self.df['Time'].dt.year == one_year_ago.year)) |
                    ((self.df['Time'].dt.month == two_years_ago.month) &
                     (self.df['Time'].dt.year == two_years_ago.year))
                ]['Time'].dt.date.unique()
                
                print("Available dates in these months/years:")
                print(sorted(nearby_dates))
                
                # Try to find the closest available dates
                available_dates = pd.to_datetime(nearby_dates)
                if len(available_dates) > 0:
                    closest_dates = []
                    for prev_date in [one_year_ago, two_years_ago]:
                        closest = min(available_dates, key=lambda x: abs(x - prev_date))
                        closest_dates.append(closest)
                    
                    historical_data = self.df[
                        self.df['Time'].dt.date.isin([d.date() for d in closest_dates])
                    ].copy()
            
            if not historical_data.empty:
                print(f"\nFound {len(historical_data)} records")
                print("Sample of historical data:")
                print(historical_data[['Time', 'Hour', 'Workable']].head())
            
            return historical_data
            
        except Exception as e:
            print(f"Error in get_historical_data: {str(e)}")
            return pd.DataFrame()
    
    
    def get_dayofweek_average(self, target_date):
        """Get average for that day of week"""
        try:
            target_day = target_date.day_name()
            dow_data = self.df[self.df['DayOfWeek'] == target_day]
            
            # Calculate average by hour
            dow_avg = dow_data.groupby('Hour')['Workable'].mean()
            
            print(f"\nDay of week ({target_day}) averages:")
            print(dow_avg)
            
            return dow_avg
        except Exception as e:
            print(f"Error in get_dayofweek_average: {str(e)}")
            return pd.Series([0] * 24, index=range(24))
    
    def get_month_average(self, target_date):
        """Get average for that month"""
        try:
            target_month = target_date.month
            month_data = self.df[self.df['Month'] == target_month]
            
            # Calculate average by hour
            month_avg = month_data.groupby('Hour')['Workable'].mean()
            
            print(f"\nMonth ({target_month}) averages:")
            print(month_avg)
            
            return month_avg
        except Exception as e:
            print(f"Error in get_month_average: {str(e)}")
            return pd.Series([0] * 24, index=range(24))
    
    
    def train_ml_model(self):
        """Train the machine learning model"""
        try:
            # Prepare features for ML
            features = ['Hour', 'Month', 'WeekOfYear', 'IsWeekend', 
                    'IsHoliday', 'IsNearHoliday', 'IsPeakHour']
            
            # Verify data
            print("\nFeature verification:")
            for feature in features:
                print(f"{feature} - Unique values: {self.df[feature].nunique()}")
                print(f"{feature} - NaN values: {self.df[feature].isna().sum()}")
            
            # One-hot encode categorical variables
            categorical_features = ['DayOfWeek', 'Season']
            print("\nCategorical features verification:")
            for feature in categorical_features:
                print(f"{feature} - Unique values: {self.df[feature].unique()}")
            
            self.encoder = OneHotEncoder(sparse_output=False)
            encoded_cats = self.encoder.fit_transform(self.df[categorical_features])
            
            # Combine all features
            X = np.hstack([
                self.df[features].values,
                encoded_cats
            ])
            y = self.df['Workable'].values
            
            print("\nTraining data shape:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            
            # Train Random Forest model
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42,
                bootstrap=True,
                oob_score=True
            )
            self.model.fit(X, y)
            
            # Store feature names
            self.feature_names = (
                features + 
                [f"{feat}_{cat}" for feat, cats in zip(categorical_features, 
                self.encoder.categories_) for cat in cats]
            )
            print("fire - ", self.model.oob_score_)
            print("\nModel training completed successfully")
            
        except Exception as e:
            print(f"Error in train_ml_model: {str(e)}")
            print("DataFrame info:")
            print(self.df.info())
            raise
        
        
    def _get_base_predictions(self, target_date):
        """Get traditional predictions based on historical data"""
        try:
            # Get historical data
            historical = self.get_historical_data(target_date)
            dow_avg = self.get_dayofweek_average(target_date)
            month_avg = self.get_month_average(target_date)
            
            hours = range(24)
            predictions = []
            historical_values = []
            
            print("\nCalculating base predictions:")
            for hour in hours:
                # Get historical value for this hour
                if not historical.empty:
                    hour_data = historical[historical['Hour'] == hour]
                    if not hour_data.empty:
                        hist_val = hour_data['Workable'].iloc[0]
                    else:
                        hist_val = (dow_avg.get(hour, 0) + month_avg.get(hour, 0)) / 2
                else:
                    hist_val = (dow_avg.get(hour, 0) + month_avg.get(hour, 0)) / 2
                
                historical_values.append(hist_val)
                
                # Calculate weighted prediction
                weighted_pred = (hist_val * 0.65) + \
                              (dow_avg.get(hour, 0) * 0.20) + \
                              (month_avg.get(hour, 0) * 0.15)
                
                print(f"\nHour {hour}:")
                print(f"  Historical: {hist_val:.2f}")
                print(f"  Day of Week Avg: {dow_avg.get(hour, 0):.2f}")
                print(f"  Month Avg: {month_avg.get(hour, 0):.2f}")
                print(f"  Weighted Prediction: {weighted_pred:.2f}")
                
                predictions.append(weighted_pred)
            
            return hours, predictions, historical_values
            
        except Exception as e:
            print(f"Error in _get_base_predictions: {str(e)}")
            return range(24), [0] * 24, [0] * 24
    
    
    def make_prediction(self, target_date_str):
        """Make predictions using both ML and traditional methods"""
        try:
            target_date = pd.to_datetime(target_date_str)
            
            # Get traditional predictions
            hours, base_predictions, historical_values = self._get_base_predictions(target_date)
            
            # Get ML predictions
            ml_predictions = self._get_ml_predictions(target_date)
            
            # Combine predictions (70% ML, 30% traditional)
            final_predictions = []
            for hour in range(24):
                ml_pred = ml_predictions[hour]
                base_pred = base_predictions[hour]
                
                # Debug print
                print(f"\nHour {hour}:")
                print(f"ML Prediction: {ml_pred:.2f}")
                print(f"Base Prediction: {base_pred:.2f}")
                
                combined_pred = (ml_pred * 0.37) + (base_pred * 0.63)
                final_predictions.append(combined_pred)
                print(f"Combined Prediction: {combined_pred:.2f}")
            
            return hours, final_predictions, historical_values, ml_predictions
            
        except Exception as e:
            print(f"Error in make_prediction: {str(e)}")
            return range(24), [0] * 24, [0] * 24, [0] * 24

    def _get_ml_predictions(self, target_date):
        predictions = []
        for hour in range(24):
            # Create feature vector for this hour
            features = pd.DataFrame({
                'Hour': [hour],
                'Month': [target_date.month],
                'WeekOfYear': [target_date.isocalendar()[1]],
                'IsWeekend': [target_date.weekday() >= 5],
                'IsHoliday': [target_date.date() in us_holidays()],
                'IsNearHoliday': [self._is_near_holiday({'Time': target_date})],
                'IsPeakHour': [1 if (9 <= hour <= 17) or (6 <= hour <= 8) else 0],
                'DayOfWeek': [target_date.day_name()],
                'Season': [self._get_season(target_date.month)]
            })
            
            # Encode categorical features
            encoded_cats = self.encoder.transform(
                features[['DayOfWeek', 'Season']]
            )
            
            # Combine features
            X = np.hstack([
                features[['Hour', 'Month', 'WeekOfYear', 'IsWeekend',
                         'IsHoliday', 'IsNearHoliday', 'IsPeakHour']].values,
                encoded_cats
            ])
            
            # Make prediction
            pred = self.model.predict(X)[0]
            predictions.append(pred)
            
        return predictions

    def plot_prediction(self, target_date_str):
        """Plot predictions and feature importance"""
        try:
            target_date = pd.to_datetime(target_date_str)
            hours, predictions, historical_values, ml_predictions = self.make_prediction(target_date_str)
    
            # Get data from two years ago
            two_years_ago = target_date - pd.DateOffset(years=2)
            historical_data_2yrs = self.df[
                (self.df['Time'].dt.month == two_years_ago.month) & 
                (self.df['Time'].dt.day == two_years_ago.day) &
                (self.df['Time'].dt.year == two_years_ago.year)
            ]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
            
            # Plot predictions and historical data
            ax1.plot(hours, predictions, 'b-', label='Combined Prediction', linewidth=2)
            ax1.plot(hours, historical_values, 'r--', 
                    label=f'Previous Year ({target_date.year-1})', alpha=0.8)
            
            # Add two years ago data if available
            if not historical_data_2yrs.empty:
                two_years_values = historical_data_2yrs.sort_values('Hour')['Workable'].values
                ax1.plot(hours, two_years_values, 'g--', 
                        label=f'Two Years Ago ({target_date.year-2})', alpha=0.8)
            
            ax1.plot(hours, ml_predictions, 'y--', label='ML Prediction', alpha=0.8)
            
            ax1.set_title(f'Charge Prediction for {target_date_str}')
            ax1.set_xlabel('Hour of Day')
            ax1.set_ylabel('Predicted Charge')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot feature importance
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            ax2.bar(range(len(importances)), importances[indices])
            ax2.set_title('Feature Importance')
            ax2.set_xticks(range(len(importances)))
            ax2.set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                'Hour': hours,
                'Combined_Prediction': predictions,
                'ML_Prediction': ml_predictions,
                'Previous_Year': historical_values
            })
            
            # Add two years ago data to results if available
            if not historical_data_2yrs.empty:
                results_df['Two_Years_Ago'] = two_years_values
            
            return results_df
            
        except Exception as e:
            print(f"Error in plot_prediction: {str(e)}")
            return pd.DataFrame()
    

    def analyze_feature_importance(self):
        """Analyze and display feature importance"""
        importances = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return feature_imp

class ChargePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Charge Predictor")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        try:
            # Initialize the predictor
            self.predictor = ChargePredictor(r"C:\Users\tterrag\.vscode\SMF1trials\CHARGE\combine.csv")
            
            # Create main frames
            self.left_frame = ttk.Frame(root, padding="10")
            self.left_frame.grid(row=0, column=0, sticky="nsew")
            
            self.right_frame = ttk.Frame(root, padding="10")
            self.right_frame.grid(row=0, column=1, sticky="nsew")
            
            # Configure grid weights
            root.grid_columnconfigure(1, weight=1)
            root.grid_rowconfigure(0, weight=1)
            
            self.setup_date_selector()
            self.setup_results_area()
            
        except Exception as e:
            tk.messagebox.showerror("Initialization Error", f"Error initializing application: {str(e)}")
            self.root.destroy()
    
    def on_closing(self):
        """Handle window closing"""
        if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
            plt.close('all')  # Close all matplotlib figures
            self.root.destroy()
            sys.exit(0)
            
    def setup_date_selector(self):
        # Date Selection Frame
        date_frame = ttk.LabelFrame(self.left_frame, text="Select Date", padding="10")
        date_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        # Calendar with no date restrictions
        today = datetime.now()
        self.cal = Calendar(date_frame, 
                           selectmode='day',
                           date_pattern='y-mm-dd',
                           year=today.year,
                           month=today.month,
                           day=today.day)
        self.cal.grid(row=0, column=0, padx=5, pady=5)
        
        # Predict Button
        ttk.Button(date_frame, text="Generate Prediction", 
                  command=self.generate_prediction).grid(row=1, column=0, pady=10)
    
        
    def setup_results_area(self):
        # Create the results frame
        results_frame = ttk.Frame(self.right_frame)
        results_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights for proper resizing
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        # Create notebook
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Graph tab (adding navigation toolbar for pan/zoom)
        self.graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.graph_frame, text="Graphs")
        
        # Split predictions into separate tabs
        self.combined_frame = ttk.Frame(self.notebook)
        self.ml_frame = ttk.Frame(self.notebook)
        self.previous_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.combined_frame, text="Combined Predictions")
        self.notebook.add(self.ml_frame, text="ML Predictions")
        self.notebook.add(self.previous_frame, text="Previous Year")
        
        # Create scrolled text widgets for each prediction type
        self.combined_text = self.create_scrolled_text(self.combined_frame)
        self.ml_text = self.create_scrolled_text(self.ml_frame)
        self.previous_text = self.create_scrolled_text(self.previous_frame)
    
        
        
    def create_scrolled_text(self, parent):
        text_widget = tk.Text(parent, height=30, width=50)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        return text_widget
    
    def generate_prediction(self):
        # Clear previous results
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        self.combined_text.delete(1.0, tk.END)
        self.ml_text.delete(1.0, tk.END)
        self.previous_text.delete(1.0, tk.END)
        
        # Get selected date
        selected_date = self.cal.get_date()
        
        # Generate predictions
        results_df = self.predictor.plot_prediction(selected_date)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # Plot predictions
        results_df.plot(x='Hour', y=['Combined_Prediction', 'ML_Prediction', 'Previous_Year'], 
                       ax=ax1, marker='o')
        ax1.set_title(f'Predictions for {selected_date}')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Predicted Charge')
        ax1.grid(True)
        
        # Plot feature importance
        importances = self.predictor.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        ax2.bar(range(len(importances)), importances[indices])
        ax2.set_title('Feature Importance')
        ax2.set_xticks(range(len(importances)))
        ax2.set_xticklabels([self.predictor.feature_names[i] for i in indices], 
                           rotation=45, ha='right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Add figure to GUI with navigation toolbar
        canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
        canvas.draw()
        
        # Add toolbar for pan/zoom functionality
        toolbar = NavigationToolbar2Tk(canvas, self.graph_frame)
        toolbar.update()
        
        # Pack canvas and toolbar
        toolbar.grid(row=0, column=0, sticky="ew")
        canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
        # Enable hover tooltips
        self.setup_hover_tooltips(fig, ax1, results_df)
        
        # Display predictions in separate tabs
        self.display_predictions(results_df, selected_date)
    
    def setup_hover_tooltips(self, fig, ax, df):
        def on_hover(event):
            if event.inaxes == ax:
                for line in ax.get_lines():
                    cont, ind = line.contains(event)
                    if cont:
                        x = line.get_xdata()[ind["ind"][0]]
                        y = line.get_ydata()[ind["ind"][0]]
                        label = line.get_label()
                        ax.annotate(f'{label}\nHour: {int(x)}\nValue: {y:.2f}',
                                  xy=(x, y), xytext=(10, 10),
                                  textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                  arrowprops=dict(arrowstyle='->'))
                        fig.canvas.draw_idle()
                        return
        
        def on_leave(event):
            if event.inaxes == ax:
                ax.clear()
                df.plot(x='Hour', y=['Combined_Prediction', 'ML_Prediction', 'Previous_Year'], 
                       ax=ax, marker='o')
                ax.set_title(f'Predictions for {self.cal.get_date()}')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Predicted Charge')
                ax.grid(True)
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        fig.canvas.mpl_connect('axes_leave_event', on_leave)
    
    def display_predictions(self, results_df, selected_date):
        # Display in Combined tab
        self.combined_text.insert(tk.END, f"Combined Predictions for {selected_date}\n")
        self.combined_text.insert(tk.END, "-" * 40 + "\n\n")
        for _, row in results_df.iterrows():
            hour = int(row['Hour'])
            pred = row['Combined_Prediction']
            self.combined_text.insert(tk.END, f"Hour {hour:02d}:00 - {pred:,.2f}\n")
        
        # Display in ML tab
        self.ml_text.insert(tk.END, f"ML Predictions for {selected_date}\n")
        self.ml_text.insert(tk.END, "-" * 40 + "\n\n")
        for _, row in results_df.iterrows():
            hour = int(row['Hour'])
            pred = row['ML_Prediction']
            self.ml_text.insert(tk.END, f"Hour {hour:02d}:00 - {pred:,.2f}\n")
        
        # Display in Previous Year tab
        self.previous_text.insert(tk.END, f"Previous Year Values for {selected_date}\n")
        self.previous_text.insert(tk.END, "-" * 40 + "\n\n")
        for _, row in results_df.iterrows():
            hour = int(row['Hour'])
            pred = row['Previous_Year']
            self.previous_text.insert(tk.END, f"Hour {hour:02d}:00 - {pred:,.2f}\n")

def main():
    root = tk.Tk()
    app = ChargePredictorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
