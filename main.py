from src.preprocessing import load_data, clean_data, scale_features
from src.features import create_features
from src.model import train_model
from src.predict import predict_failure
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = load_data("data/sensor_data.csv")

# Clean
df = clean_data(df)

# Feature engineering
df = create_features(df)

# Features & target
X = df[["temperature", "vibration", "pressure", "rpm",
        "temp_roll_mean", "vib_roll_mean", "rpm_roll_mean"]]

y = df["failure"]

# Scale
X, scaler = scale_features(X, X.columns)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = train_model(X_train, y_train)

# Predict
y_pred = predict_failure(model, X_test)

# Evaluate
print(classification_report(y_test, y_pred))
from src.visualize import plot_sensor_data

plot_sensor_data(df)