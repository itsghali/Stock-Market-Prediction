# Stock-Market-Prediction

#Overview
This web application is designed to predict stock prices using linear and polynomial regression models. It allows users to analyze historical stock data, visualize predictions, and evaluate model performance. The application also includes a user authentication system and displays news headlines related to stock markets. The application is built using Streamlit, SQLite, and a combination of data science libraries such as scikit-learn, yfinance, Selenium, and BeautifulSoup.

Features
Stock Price Prediction Models:

Linear regression and polynomial regression models are used to predict stock prices.

Visualize predicted stock prices alongside actual data.

User Authentication:

Users can sign up and log in using SQLite for data storage.

Passwords are securely hashed using SHA-256 for better security.

Stock Data Fetching:

Historical stock data is fetched using yfinance (Yahoo Finance).

Moving Averages:

Calculate and display different moving averages (50, 100, 200) alongside stock prices.

Stock Market News:

Fetch and display the latest stock market news headlines using Selenium and BeautifulSoup.

Model Evaluation:

Evaluate prediction models using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.

Requirements
To run the application, the following dependencies must be installed:

Python 3.x

Required Python packages (installable via pip):

bash
Copy
Edit
pip install numpy pandas matplotlib yfinance streamlit sqlite3 hashlib scikit-learn selenium webdriver-manager beautifulsoup4
Chrome WebDriver for Selenium (automatically managed by webdriver-manager).

Installation
Clone the repository or download the code files.

Install dependencies by running the following command in your terminal:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
streamlit run app.py
This will start the application and open it in your default web browser.

Code Breakdown
1. Database Functions
create_connection: Establishes a connection to the SQLite database.

create_users_table: Creates a table for storing user information (username, password).

add_user: Adds a new user to the database.

user_exists: Checks if a user already exists.

validate_user: Validates login credentials.

2. Security
hash_password: Hashes the password using SHA-256 for secure storage.

3. Stock Functions
get_tickers: Scrapes the list of tickers (companies) from Wikipedia pages for S&P 500 and NASDAQ-100 indices.

load_stock_data: Fetches historical stock data using yfinance and computes moving averages (50, 100, 200).

train_prediction_models: Trains both linear and polynomial regression models using historical data.

make_predictions: Uses the trained models to make predictions for future stock prices.

plot_stock_analysis: Plots stock prices and moving averages, along with the model predictions.

fetch_overall_news: Uses Selenium and BeautifulSoup to fetch the latest stock market news headlines.

evaluate_model: Evaluates the model's performance using MAE, MSE, RMSE, and R² score.

interpret_metrics: Interprets the evaluation metrics for model quality assessment.

evaluate_models_and_display: Displays the results of model evaluation in the web app.

determine_better_model: Compares the linear and polynomial models to determine which one performs better based on evaluation metrics.

Web Interface
The web interface is built with Streamlit. The main components include:

Login/Signup Form: A user can create an account and log in using the authentication system.

Stock Data Display: Displays stock data along with moving averages and predictions.

Model Evaluation: Shows evaluation metrics and the interpretation of prediction errors.

Stock Market News: Displays the latest stock market news.

User Authentication
Signup: When a new user signs up, their username and password are stored in the SQLite database. Passwords are hashed before storage.

Login: Users can log in with their credentials, which are validated against the database.

Model Evaluation Metrics
The models are evaluated using the following metrics:

Mean Absolute Error (MAE): The average absolute difference between the predicted and actual values.

Mean Squared Error (MSE): The average of the squared differences between predicted and actual values.

Root Mean Squared Error (RMSE): The square root of the MSE, providing an estimate of the standard deviation of the prediction error.

R² Score: The proportion of the variance in the dependent variable that is predictable from the independent variable(s).

Conclusion
This stock prediction web application allows users to explore and visualize stock predictions using machine learning models. The interactive Streamlit interface and robust evaluation metrics make it an insightful tool for analyzing stock data.

For further improvements, consider integrating more advanced machine learning models, real-time data feeds, and a more secure authentication system.
