import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime, timedelta
import re

# Streamlit app configuration
st.set_page_config(page_title="Automated Sales Forecast with LightGBM", layout="wide")

# Function to clean feature names
def clean_feature_name(name):
    name = str(name)
    name = re.sub(r'[^\w\s]', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = name.strip('_')
    if not name or name[0].isdigit():
        name = f"feature_{name}"
    return name

# Title and description
st.title("Automated Sales Forecast with LightGBM")
st.write("Upload a CSV file with a date column and a sales column. The app will automatically detect and process the data.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # List of encodings to try
    encodings = ['utf-8', 'latin1', 'windows-1252', 'iso-8859-1', 'utf-16']
    df = None
    
    # Try reading the CSV with each encoding
    for encoding in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            st.write(f"Successfully read CSV")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error reading file with {encoding} encoding: {str(e)}")
            continue
    
    if df is None:
        st.error("Failed to read CSV with any supported encoding. Please ensure the file is a valid CSV with a compatible encoding (e.g., UTF-8, Latin1, Windows-1252).")
        st.stop()
    
    try:
        # Clean column names
        df.columns = [clean_feature_name(col) for col in df.columns]
        
        # Automatically detect date and sales columns
        date_candidates = [col for col in df.columns if any(x in col.lower() for x in ['date', 'day', 'time'])]
        sales_candidates = [col for col in df.columns if any(x in col.lower() for x in ['sales', 'revenue', 'amount']) and pd.api.types.is_numeric_dtype(df[col])]
        
        if not date_candidates:
            st.error("No date column detected. Please ensure your CSV has a column named 'date', 'Date', 'day', or similar.")
            st.stop()
        if not sales_candidates:
            st.error("No sales column detected. Please ensure your CSV has a column named 'sales', 'Sales', 'revenue', or similar with numeric values.")
            st.stop()
        
        date_col = date_candidates[0]
        sales_col = sales_candidates[0]
        # st.write(f"Automatically detected date column: {date_col}")
        # st.write(f"Automatically detected sales column: {sales_col}")
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isna().any():
            st.error(f"Invalid date format in '{date_col}' column. Ensure dates are in a valid format (e.g., YYYY-MM-DD).")
            st.stop()
        
        # Ensure sales column is numeric
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
        if df[sales_col].isna().any():
            st.error(f"Invalid values in '{sales_col}' column. Ensure all values are numeric.")
            st.stop()
        
        # Keep all duplicate rows for modeling
        # st.write("All data points, including duplicates, are used for prediction to improve accuracy.")
        # st.write("Feature selection is performed using LightGBM importance, selecting the top 80% cumulative importance features for optimal performance.")
        
        # Enhanced Feature Engineering
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['lag_1'] = df[sales_col].shift(1)
        df['lag_7'] = df[sales_col].shift(7)
        df['rolling_mean_7'] = df[sales_col].rolling(window=7).mean()
        df['ema_7'] = df[sales_col].ewm(span=7, adjust=False).mean()
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['week_of_year'] = df[date_col].dt.isocalendar().week
        df['season'] = df[date_col].dt.month % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
        # Add linear trend based on the last 30 days
        trend_window = min(30, len(df))
        df['trend'] = df[sales_col].rolling(window=trend_window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] * len(x) + np.polyfit(range(len(x)), x, 1)[1], raw=True).shift(1)
        
        # Drop rows with NaN values (due to lagging)
        df = df.dropna()
        
        # Identify all other columns as features
        feature_cols = [col for col in df.columns if col not in [date_col, sales_col]]
        categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Encode categorical variables for training data
        df_encoded = pd.get_dummies(df[feature_cols], columns=categorical_cols)
        df_encoded.columns = [clean_feature_name(col) for col in df_encoded.columns]
        
        # Prepare X and y
        X = df_encoded
        y = df[sales_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Initial LightGBM model for feature selection
        initial_model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, max_depth=20, random_state=42)
        initial_model.fit(X_train, y_train)
        
        # Feature selection based on importance
        importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': initial_model.feature_importances_})
        importance = importance.sort_values('Importance', ascending=False)
        importance['Cumulative'] = importance['Importance'].cumsum() / importance['Importance'].sum()
        selected_features = importance[importance['Cumulative'] <= 0.8]['Feature'].tolist()
        if not selected_features:
            selected_features = importance['Feature'].tolist()
        # st.write(f"Selected {len(selected_features)} out of {len(X_train.columns)} features based on importance.")
        
        # Update X_train and X_test with selected features
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
        
        # Train final LightGBM model
        model = LGBMRegressor(n_estimators=2000, learning_rate=0.05, max_depth=20, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = np.array(model.predict(X_test))
        
        # Calculate custom accuracy (100 - MAPE, capped at 0%)
        def calculate_mape(y_true, y_pred):
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        mape = calculate_mape(y_test, y_pred)
        accuracy = max(0, 100 - mape)  # Cap at 0% to avoid negatives
        
        # Display raw uploaded data
        st.subheader("Raw Uploaded Sales Data")
        st.write(df.head())  # Show raw data as uploaded
        
        # Display accuracy
        st.subheader("Model Performance")
        st.metric("Accuracy", f"{accuracy:.2f}%")
        
        # Forecasting future sales
        st.subheader("Sales Forecast")
        periods = 90  # Default to 90 days, automated
        st.write(f"Forecasting for {periods} days.")
        
        # Create future dates
        last_date = df[date_col].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        future_df = pd.DataFrame({date_col: future_dates})
        
        # Add time-based features with increased random variation
        future_df['year'] = future_df[date_col].dt.year
        future_df['month'] = future_df[date_col].dt.month
        future_df['day'] = future_df[date_col].dt.day
        future_df['dayofweek'] = future_df[date_col].dt.dayofweek
        future_df['quarter'] = future_df[date_col].dt.quarter
        future_df['lag_1'] = float(df[sales_col].iloc[-1]) * (1 + np.random.normal(0, 0.1))
        future_df['lag_7'] = float(df[sales_col].iloc[-7]) * (1 + np.random.normal(0, 0.1)) if len(df) >= 7 else float(df[sales_col].iloc[-1]) * (1 + np.random.normal(0, 0.1))
        future_df['rolling_mean_7'] = float(df[sales_col].iloc[-7:].mean()) * (1 + np.random.normal(0, 0.1)) if len(df) >= 7 else float(df[sales_col].mean()) * (1 + np.random.normal(0, 0.1))
        future_df['ema_7'] = float(df[sales_col].iloc[-1]) * (1 + np.random.normal(0, 0.1))
        future_df['is_month_start'] = future_df[date_col].dt.is_month_start.astype(int)
        future_df['is_month_end'] = future_df[date_col].dt.is_month_end.astype(int)
        future_df['week_of_year'] = future_df[date_col].dt.isocalendar().week
        future_df['season'] = future_df[date_col].dt.month % 12 // 3 + 1
        # Add trend based on the last trend value
        last_trend = float(df['trend'].iloc[-1])
        future_df['trend'] = last_trend + np.linspace(0, last_trend * 0.1, periods)
        
        # Handle other features (use most recent values with variation for numeric, last value for categorical)
        for col in numerical_cols:
            future_df[col] = float(df[col].iloc[-1]) * (1 + np.random.normal(0, 0.1))
        for col in categorical_cols:
            future_df[col] = df[col].iloc[-1]
        
        # Encode future data and align with training features
        future_df_encoded = pd.get_dummies(future_df[feature_cols], columns=categorical_cols)
        future_df_encoded.columns = [clean_feature_name(col) for col in future_df_encoded.columns]
        future_df_encoded = future_df_encoded.reindex(columns=selected_features, fill_value=0)
        
        # Predict future sales
        future_pred = np.array(model.predict(future_df_encoded))
        # Add confidence band (e.g., ±20% for more variability)
        future_pred_lower = future_pred * 0.8
        future_pred_upper = future_pred * 1.2
        
        # Aggregate historical data for plotting
        df_plot = df.groupby(date_col).agg({sales_col: 'mean'}).reset_index()
        
        # Plot forecast with line and shaded region
        st.subheader("Sales Forecast")
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df_plot[date_col], y=df_plot[sales_col], name='Historical Sales', line=dict(color='#27ae60', width=2), mode='lines+markers', marker=dict(size=6)))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_pred, name='Forecasted Sales', line=dict(color='#3498db', width=2), mode='lines+markers', marker=dict(size=6)))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_pred_upper, line=dict(color='rgba(52, 152, 219, 0.2)', width=0), fill='tonexty', showlegend=False))
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_pred_lower, line=dict(color='rgba(52, 152, 219, 0.2)', width=0), fill='tonexty', name='Confidence Band', showlegend=True))
        fig_forecast.update_layout(
            title_text="Sales Forecast with Confidence Band",
            title_font_size=20,
            xaxis_title="Date",
            yaxis_title="Sales",
            template="plotly_white",
            xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', tickangle=45),
            yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)'),
            legend=dict(font_size=12, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=400
        )
        fig_forecast.update_traces(hovertemplate="%{x}<br>Sales: %{y}<extra></extra>")
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Download forecast data
        forecast_df = pd.DataFrame({date_col: future_dates, 'yhat': future_pred})
        forecast_csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast Data",
            data=forecast_csv,
            file_name="sales_forecast_lightgbm.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis.")