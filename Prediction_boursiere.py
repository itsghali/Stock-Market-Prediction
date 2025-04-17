import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
import sqlite3
import hashlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from datetime import datetime
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from streamlit_chat import message

# ---------- DB FUNCTIONS ----------
# Connect to SQLite database
def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

# Create users table
def create_users_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Add a new user
def add_user(username, password):
    conn = create_connection()
    c = conn.cursor()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()

# Check if user exists
def user_exists(username):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    return result is not None

# Validate login
def validate_user(username, password):
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

# ---------- SECURITY ----------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------- STOCK FUNCTIONS ----------
def get_tickers():
    url = [
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'https://en.wikipedia.org/wiki/NASDAQ-100',
    ]

    tickers = []

    for u in url:
        tables = pd.read_html(u)

        if len(tables) > 0:
            table = tables[0]
            if 'Symbol' in table.columns:
                tickers += table['Symbol'].tolist()

    return tickers

def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date,end=end_date)
    df['Days'] = np.arange(len(df))
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['100_MA'] = df['Close'].rolling(window=100).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()
    return df

def train_prediction_models(df):
    X = df['Days'].values.reshape(-1, 1)
    linear_model = LinearRegression()
    linear_model.fit(X, df['Close'].values)
    poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_model.fit(X, df['Close'].values)
    return linear_model, poly_model

def make_predictions(df, linear_model, poly_model, future_years):
    future_days = future_years * 365  # Convert years to days (approximation)
    X_future = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    linear_predictions = linear_model.predict(X_future)
    poly_predictions = poly_model.predict(X_future)
    
    # Generate future dates
    future_dates = pd.date_range(start=df.index[-1], periods=future_days + 1, freq="D")[1:]
    
    predictions_df = pd.DataFrame({
        "Date": future_dates,
        "Linear Prediction": linear_predictions.flatten(),
        "Polynomial Prediction": poly_predictions.flatten()
    })
    return predictions_df

def plot_stock_analysis(df, predictions_df, show_mas, prediction_type):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-3650:], df['Close'].values[-3650:],
             label="Prix réels", color="blue", linewidth=2)
    if show_mas:
        plt.plot(df.index[-3650:], df['50_MA'].values[-3650:],
                 label="50MA", color="yellow", linewidth=2, linestyle='--')
        plt.plot(df.index[-3650:], df['100_MA'].values[-3650:],
                 label="100MA", color="orange", linewidth=2, linestyle='--')
        plt.plot(df.index[-3650:], df['200_MA'].values[-3650:],
                 label="200MA", color="red", linestyle='--')
    if prediction_type == "Linear":
        plt.plot(predictions_df['Date'], predictions_df['Linear Prediction'],
                 label="Prédiction Linéaire", color="green", linestyle="--", linewidth=2)
    elif prediction_type == "Polynomial":
        plt.plot(predictions_df['Date'], predictions_df['Polynomial Prediction'],
                 label="Prédiction Polynomiale", color="purple", linestyle="--", linewidth=2)
    plt.scatter(df.index[-1], df['Close'].values[-1], color='red', s=100, zorder=5,
                label="Dernier Prix")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Prix", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    return plt

def fetch_overall_news():
    url = "https://finance.yahoo.com/topic/stock-market-news/"

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(url)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    articles = soup.find_all("h3")
    if articles:
        headline = articles[0].text
        link = "https://finance.yahoo.com/topic/stock-market-news/"
        return {"headline": headline, "link": link}

    return {"headline": "No news available", "link": ""}

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)
    return mae, mse, rmse, r2

def interpret_metrics(mae, mse, rmse, r2, current_price):
    interpretations = {}

    if isinstance(mae, pd.Series):
        mae = mae.iloc[0]
    if isinstance(current_price, pd.Series):
        current_price = current_price.iloc[0]
    if isinstance(rmse, pd.Series):
        rmse = rmse.iloc[0]
    if isinstance(r2, pd.Series):
        r2 = r2.iloc[0]
    if isinstance(mse, pd.Series):
        mse = mse.iloc[0]

    # Calculate percentage errors relative to the current price
    price_percentage_mae = (mae / current_price) * 100
    price_percentage_rmse = (rmse / current_price) * 100
    price_percentage_mse = (mse / current_price) * 100

    if price_percentage_mae < 1:
        mae_quality = "excellent"
        mae_interpretation = f"predictions are off by only ${mae:.2f} on average (less than 1% of stock price)"
    elif price_percentage_mae < 3:
        mae_quality = "good"
        mae_interpretation = f"predictions are off by ${mae:.2f} on average (about {price_percentage_mae:.1f}% of stock price)"
    elif price_percentage_mae < 7:
        mae_quality = "moderate"
        mae_interpretation = f"predictions are off by ${mae:.2f} on average ({price_percentage_mae:.1f}% of stock price)"
    else:
        mae_quality = "poor"
        mae_interpretation = f"predictions are off by ${mae:.2f} on average (a significant {price_percentage_mae:.1f}% of stock price)"

    interpretations["mae"] = {
        "quality": mae_quality,
        "interpretation": mae_interpretation
    }

    rmse_to_mae_ratio = rmse / mae if mae > 0 else 0

    if rmse_to_mae_ratio < 1.2:
        error_variability = "very consistent"
    elif rmse_to_mae_ratio < 1.5:
        error_variability = "fairly consistent"
    elif rmse_to_mae_ratio < 2:
        error_variability = "somewhat variable"
    else:
        error_variability = "highly variable"

    interpretations["rmse"] = {
        "ratio_to_mae": rmse_to_mae_ratio,
        "variability": error_variability,
        "interpretation": f"Prediction errors are {error_variability} (RMSE is {rmse_to_mae_ratio:.1f}x the MAE)"
    }

    if price_percentage_mse < 1:
        mse_quality = "excellent"
        mse_interpretation = f"predictions have a very small squared error of ${mse:.2f} (less than 1% of stock price)"
    elif price_percentage_mse < 3:
        mse_quality = "good"
        mse_interpretation = f"predictions have a relatively small squared error of ${mse:.2f} (about {price_percentage_mse:.1f}% of stock price)"
    elif price_percentage_mse < 7:
        mse_quality = "moderate"
        mse_interpretation = f"predictions have a moderate squared error of ${mse:.2f} (around {price_percentage_mse:.1f}% of stock price)"
    else:
        mse_quality = "poor"
        mse_interpretation = f"predictions have a significant squared error of ${mse:.2f} (a considerable {price_percentage_mse:.1f}% of stock price)"

    interpretations["mse"] = {
        "quality": mse_quality,
        "interpretation": mse_interpretation
    }

    if r2 < 0.3:
        r2_quality = "poor"
        r2_interpretation = f"model explains only {r2 * 100:.1f}% of price movements (not reliable)"
    elif r2 < 0.5:
        r2_quality = "fair"
        r2_interpretation = f"model explains {r2 * 100:.1f}% of price movements (limited reliability)"
    elif r2 < 0.7:
        r2_quality = "moderate"
        r2_interpretation = f"model explains {r2 * 100:.1f}% of price movements (moderately reliable)"
    elif r2 < 0.85:
        r2_quality = "good"
        r2_interpretation = f"model explains {r2 * 100:.1f}% of price movements (good reliability)"
    else:
        r2_quality = "excellent"
        r2_interpretation = f"model explains {r2 * 100:.1f}% of price movements (highly reliable)"

    interpretations["r2"] = {
        "quality": r2_quality,
        "interpretation": r2_interpretation
    }

    return interpretations

# Evaluate models
def evaluate_models_and_display(linear_model, poly_model, df, ticker):
    linear_metrics = evaluate_model(linear_model, df['Days'].values.reshape(-1, 1), df['Close'].values)
    poly_metrics = evaluate_model(poly_model, df['Days'].values.reshape(-1, 1), df['Close'].values)

    # Get current stock price for context
    current_price = df['Close'].iloc[-1]

    # Interpret metrics for both models
    linear_interpret = interpret_metrics(*linear_metrics, current_price)
    poly_interpret = interpret_metrics(*poly_metrics, current_price)

    # Function to display metrics and interpretation
    def display_model_results(model_name, metrics, interpret):
        st.subheader(f"{model_name} Regression Metrics:")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", f"${metrics[0]:.2f}")
            st.markdown(f"{interpret['mae']['interpretation']}")

            st.metric("RMSE", f"${metrics[2]:.2f}")
            st.markdown(f"{interpret['rmse']['interpretation']}")

        with col2:
            st.metric("MSE", f"${metrics[1]:.2f}")

            st.metric("R² Score", f"{metrics[3]:.4f}")
            st.markdown(f"{interpret['r2']['interpretation']}")

        # Overall assessment based on MAE and R²
        mae_quality = interpret['mae']['quality']
        r2_quality = interpret['r2']['quality']
        if mae_quality in ["good", "excellent"] and r2_quality in ["good", "excellent"]:
            overall_quality = "good"
        else:
            overall_quality = "limited"

        st.markdown(f"Overall assessment: The {model_name} Regression model shows {overall_quality} prediction quality.")

    # Determine the better model
    def determine_better_model(linear_metrics, poly_metrics):
        linear_mae, linear_mse, linear_rmse, linear_r2 = linear_metrics
        poly_mae, poly_mse, poly_rmse, poly_r2 = poly_metrics

        # Define thresholds for comparison
        mae_threshold = 0.05  # 5% difference
        rmse_threshold = 0.05 # 5% difference
        r2_threshold = 0.02   # 2% difference

        # Compare MAE and RMSE (lower is better)
        if linear_mae * (1 - mae_threshold) > poly_mae and linear_rmse * (1- rmse_threshold) > poly_rmse:
            mae_score = 1
            rmse_score = 1
        elif linear_mae < poly_mae  and linear_rmse < poly_rmse:
            mae_score = 0
            rmse_score = 0
        else :
            mae_score = 0.5
            rmse_score = 0.5

        # Compare R² (higher is better)
        if linear_r2 > poly_r2 * (1 + r2_threshold):
            r2_score = 0
        elif linear_r2 * (1 + r2_threshold) < poly_r2:
            r2_score = 1
        else:
            r2_score = 0.5

        # Overall model assessment
        total_score = mae_score + rmse_score + r2_score

        if total_score > 2 :
            return "Polynomial"
        elif total_score < 1:
            return "Linear"
        else:
            return "Inconclusive"

    # Display results for both models
    display_model_results("Linear", linear_metrics, linear_interpret)
    display_model_results("Polynomial", poly_metrics, poly_interpret)

    # Comparison of the models
    st.subheader("Model Comparison")

    # Determine which model is better
    better_model = determine_better_model(linear_metrics, poly_metrics)

    # Display the result
    if better_model == "Linear":
        st.write("Linear Regression model performs better than Polynomial Regression model.")
    elif better_model == "Polynomial":
        st.write("Polynomial Regression model performs better than Linear Regression model.")
    else:
        st.write("The models perform similarly based on the metrics.")

# ---------- STREAMLIT APP ----------
def main():
    # Setup page configuration
    st.set_page_config(
        page_title="Stock Analysis App",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for authentication
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None

    # Create the users table if it doesn't exist
    create_users_table()

    # Sidebar for login and registration
    authentication_status = st.session_state['authentication_status']
    username = st.session_state['username']

    if authentication_status is None:
        st.sidebar.subheader("Login / Register")
        login_register_option = st.sidebar.radio("Choose an option", ("Login", "Register"))

        if login_register_option == "Login":
            login_username = st.sidebar.text_input("Username")
            login_password = st.sidebar.text_input("Password", type="password")

            if st.sidebar.button("Login"):
                hashed_password = hash_password(login_password)
                if validate_user(login_username, hashed_password):
                    st.session_state['authentication_status'] = True
                    st.session_state['username'] = login_username
                    st.sidebar.success(f"Logged in as {login_username}")
                    st.rerun()  # Rerun to reflect login status
                else:
                    st.sidebar.error("Invalid username or password")

        else:  # Register
            register_username = st.sidebar.text_input("New Username")
            register_password = st.sidebar.text_input("New Password", type="password")

            if st.sidebar.button("Register"):
                if user_exists(register_username):
                    st.sidebar.error("Username already exists. Choose another.")
                else:
                    hashed_password = hash_password(register_password)
                    add_user(register_username, hashed_password)
                    st.sidebar.success("Registration successful. Please log in.")

    elif authentication_status:
        st.sidebar.subheader(f"Logged in as {username}")
        if st.sidebar.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.rerun() # Rerun to reflect logout status
        
        # ---------- STOCK ANALYSIS SECTION ----------
        st.title("Stock Analysis")

        # Move controls to sidebar
        with st.sidebar:
            # Stock Ticker Selection
            available_tickers = get_tickers()
            ticker = st.selectbox("Sélectionner un ticker boursier", available_tickers)

            # Date Range Selection
            start_date = st.date_input("Date de début", datetime(1980, 1, 1).date(), min_value=datetime(1970, 1, 1).date(), max_value=datetime.now().date())  # Convert to date
            end_date = st.date_input("Date de fin", datetime.now().date(), min_value=start_date, max_value=datetime.now().date())  # Convert to date

            # Validate date range
            if start_date >= end_date:
                st.error("La date de début doit être antérieure à la date de fin.")
            elif end_date > datetime.now().date():
                st.error("La date de fin ne peut pas dépasser la date actuelle.")
            else:
                # Prediction Horizon
                future_years = st.slider("Horizon de prédiction (années)", 1,10, 3)

                # Moving Averages Display Option
                show_mas = st.checkbox("Afficher les moyennes mobiles (50, 100, 200)")

                # Prediction Type Selection
                prediction_type = st.radio("Type de prédiction", ("Linear", "Polynomial"))

                # Analyser button
                analyse_clicked = st.button("Analyser")

                # Display news
                news = fetch_overall_news()
                st.write(f"### Latest news : [{news['headline']}]({news['link']})")

        # Only proceed if date range is valid and analyse button clicked
        if start_date < end_date and end_date <= datetime.now().date() and analyse_clicked:
            try:
                # Load stock data with error handling
                df = load_stock_data(ticker, start_date, end_date)

                # Train prediction models
                linear_model, poly_model = train_prediction_models(df)

                # Make predictions
                predictions_df = make_predictions(df, linear_model, poly_model, future_years)

                # Plot stock analysis with moving averages and predictions
                plt_obj = plot_stock_analysis(df, predictions_df, show_mas, prediction_type)
                st.pyplot(plt_obj)

                # Evaluate the models and display metrics
                evaluate_models_and_display(linear_model, poly_model, df, ticker)

            except Exception as e:
                st.error(f"Error processing: {e}")

    else:
        st.warning("Please log in to access the stock analysis app.")

if __name__ == "_main_":
    main()