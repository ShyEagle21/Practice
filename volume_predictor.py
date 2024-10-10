import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def train_and_save_model(csv_file_path, model_file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # One hot encode day of week
    df_onehot = pd.get_dummies(df, columns=['Day of Week'])

    # Feature selection
    features = ['yesterday_total_packages',
                'RAFT_known_shipped_pkg_count',
                'RAFT_predicted_carryover_pkg_count',
                'RAFT_predicted_total_handoff_pkg_count',
                'Day of Week_Sunday',
                'Day of Week_Monday',
                'Day of Week_Tuesday',
                'Day of Week_Wednesday',
                'Day of Week_Thursday',
                'Day of Week_Friday',
                'Day of Week_Saturday',
                'Promotion',
                'TMAX',
                'TMIN',
                'AWND',
                'PRCP',
                'SNOW']

    # Convert date columns to datetime
    df_onehot['Prediction_For_Date'] = pd.to_datetime(df['Prediction_For_Date'])

    # Split the data into training and testing sets
    X = df_onehot[features]
    y_package_count = df_onehot['Total Packages Received']

    X_train, X_test, y_package_train, y_package_test = train_test_split(X, y_package_count, test_size=0.2, random_state=42)

    # Train the model for actual package count prediction
    package_model = RandomForestRegressor(n_estimators=25, random_state=42)
    package_model.fit(X_train, y_package_train)

    # Predict and evaluate the model for actual package count
    y_package_pred = package_model.predict(X_test)
    r2 = r2_score(y_package_test, y_package_pred)
    package_mae = mean_absolute_error(y_package_test, y_package_pred)
    print(f'Package Count Prediction MAE: {package_mae}')
    print(f'Package Count R2: {r2}')

    # Save the model to a file
    joblib.dump(package_model, model_file_path)
    print(f'Model saved to {model_file_path}')


def load_model(model_file_path):
    # Load the model from the file
    model = joblib.load(model_file_path)
    return model

def predict_volume(model, feature_values):
    # Convert feature values to DataFrame
    feature_df = pd.DataFrame([feature_values])

    # Predict the total volume
    predicted_volume = model.predict(feature_df)
    predicted_volume_int = int(predicted_volume[0])
    return predicted_volume_int

