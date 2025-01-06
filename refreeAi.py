import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

# Load data
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Streamlit app
def main():
    st.title("Unified VAR System: Match Analysis and Predictions")

    # Upload datasets
    st.sidebar.header("Upload Datasets")
    matches_file = st.sidebar.file_uploader("Upload Match Outcomes Data (CSV)", type="csv")
    incidents_file = st.sidebar.file_uploader("Upload VAR Incidents Stats Data (CSV)", type="csv")
    team_stats_file = st.sidebar.file_uploader("Upload VAR Team Stats Data (CSV)", type="csv")

    if matches_file is not None:
        matches_data = load_data(matches_file)
        st.subheader("Matches Dataset Overview")
        st.write(matches_data.head())

        # Match outcome prediction functionality
        st.subheader("Match Outcome Prediction")
        try:
            matches_data = matches_data[['xg', 'sh', 'result']]
            matches_data['result_binary'] = matches_data['result'].apply(lambda x: 1 if x == 'W' else 0)
            matches_data = matches_data.drop(columns=['result'])

            # Split data into features and target
            X = matches_data[['xg', 'sh']]
            y = matches_data['result_binary']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the linear regression model
            outcome_model = LinearRegression()
            outcome_model.fit(X_train, y_train)

            # Predictions
            y_pred = outcome_model.predict(X_test)
            y_pred_class = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_class)

            st.write(f"Model Accuracy for Match Outcome Prediction: **{accuracy:.2f}**")

            # Sidebar inputs for prediction
            st.sidebar.header("Outcome Prediction: Input Features")
            xg = st.sidebar.slider("Expected Goals (xG)", float(X['xg'].min()), float(X['xg'].max()), float(X['xg'].mean()))
            sh = st.sidebar.slider("Shots", int(X['sh'].min()), int(X['sh'].max()), int(X['sh'].mean()))

            input_data = np.array([[xg, sh]])
            prediction = outcome_model.predict(input_data)
            prediction_class = "Win" if prediction > 0.5 else "Not Win"

            st.write(f"Prediction for Input: **{prediction_class}**")
        except KeyError:
            st.error("Matches dataset must contain columns: 'xg', 'sh', and 'result'.")

    if incidents_file is not None:
        incidents_data = load_data(incidents_file)
        st.subheader("VAR Incidents Dataset Overview")
        st.write(incidents_data.head())

        # VAR Incidents Analysis
        st.subheader("VAR Incidents Analysis")
        st.sidebar.header("VAR Incidents: Configure Analysis")
        target_variable = st.sidebar.selectbox("Select Target Variable", options=incidents_data.columns[1:])
        feature_variable = st.sidebar.selectbox("Select Feature Variable", options=incidents_data.columns[1:])

        if target_variable == feature_variable:
            st.warning("Feature and Target variables must be different.")
        else:
            X = incidents_data[[feature_variable]].values
            y = incidents_data[target_variable].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train linear regression model
            incidents_model = LinearRegression()
            incidents_model.fit(X_train, y_train)

            # Predictions
            y_pred = incidents_model.predict(X_test)

            # Results
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

            # Visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test, y_test, color='blue', label='Actual')
            plt.plot(X_test, y_pred, color='red', label='Prediction')
            plt.xlabel(feature_variable)
            plt.ylabel(target_variable)
            plt.title(f"{target_variable} vs {feature_variable}")
            plt.legend()
            st.pyplot(plt)

    if team_stats_file is not None:
        team_stats_data = load_data(team_stats_file)
        st.subheader("Team Stats Dataset Overview")
        st.write(team_stats_data.head())

        # Summary statistics for teams
        st.subheader("Team-Level VAR Stats")
        selected_team = st.sidebar.selectbox("Select Team", options=team_stats_data['Team'].unique())
        team_data = team_stats_data[team_stats_data['Team'] == selected_team]
        st.write(team_data)

if __name__ == "__main__":
    main()
