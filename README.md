Car Dheko: Used Car Price Prediction & ML Pipeline

Project Overview
The Car Dheko project is an end-to-end machine learning application designed to predict used car prices with high precision. 
It transforms unstructured, city-wise car data into a structured format, performs advanced feature engineering, 
and provides a real-time prediction interface via Streamlit.

Tech Stack
Data Processing: Python, Pandas, NumPy, Regex (Regular Expressions).
Machine Learning: Scikit-Learn (Random Forest, Gradient Boosting, Linear Regression).
Web Framework: Streamlit.
Database: MySQL (SQLAlchemy, mysql-connector-python).
Serialization: Joblib.

Advanced Features in this Implementation
1. Automated ETL Pipeline
Unlike standard datasets, this project handles nested JSON-like structures embedded in Excel sheets.
JSON Flattening: Uses ast.literal_eval and custom recursive functions to flatten "Top", "Data", and "Specs" dictionaries into separate columns.
City Consolidation: Automatically merges data from 6 major Indian cities (Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata) while maintaining city-level origin features.

2. Domain-Specific Unit Conversion
The code includes a sophisticated cleaning engine to standardize automotive metrics:
Price: Converts "Lakh" and "Crore" strings into numerical INR values.
Mileage: Standardizes kmpl and km/kg (converting gas-based fuel efficiency to a comparable liquid scale).
Power & Torque: Normalizes units across bhp, ps, kw, and kgm to standard bhp and nm units.
Physical Dimensions: Strips text units (mm, kg, meters) and converts them to floating-point numbers.

3. Feature Engineering & Quality Control
Derived Metrics: Created features like CarAge, Kms_per_year, and Power_per_CC to capture vehicle wear and performance density.
Outlier Clipping: Uses a 1st and 99th percentile clipping method to handle extreme values without losing data.
Binary Encoding: Converts various car amenities (Heater, Bluetooth, Airbags) into binary flags for model compatibility.

Machine Learning Workflow
1. Data Cleaning
Imputation: Strategic filling of missing values using Median for skewed numerical data and Mode for categorical data.
Constraint Filtering: Removes physically impossible records (e.g., cars with 0 Engine CC or negative Kms).

2. Model Selection
The project evaluates several regression algorithms:
Linear Regression (Baseline)
Decision Tree Regressor
Random Forest Regressor (Ensemble)
Gradient Boosting Regressor (Boosting)

3. Hyperparameter Tuning
Uses GridSearchCV to find the optimal parameters for the Random Forest and Gradient Boosting models,
ensuring the highest R2 score and lowest Mean Absolute Error (MAE).

Streamlit Application
The UI is designed for both technical sales reps and non-technical customers:
Interactive Sidebar: Filter by City, Body Type, and Brand (OEM).
Real-time Prediction: Uses a pre-trained Pipeline (StandardScaler + Regressor) to provide instant price estimates based on user inputs.
Database Logging: (Optional) Integration with MySQL to store prediction history.











