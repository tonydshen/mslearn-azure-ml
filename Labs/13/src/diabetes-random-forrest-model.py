# train.py
# import necessary libraries
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# get parameters
parser = argparse.ArgumentParser("train")
parser.add_argument("--training_data", type=str, help="Path to training data")
parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest")
parser.add_argument("--model_output", type=str, help="Path of output model")
parser.add_argument("--test_size", type=float, default=0.30, help="test size")
parser.add_argument("--random_state", type=int, default=None, help="random state")

args = parser.parse_args()
print(args)

training_data = args.training_data
model_output = args.model_output
n_estimators = args.n_estimators
test_size = args.test_size
random_state = args.random_state

# Load your dataset
df = pd.read_csv(training_data)  # Replace with your actual file

# Assume the last column is the target
# X = data.iloc[:, :-1]  # Features
# y = data.iloc[:, -1]   # Target

# Separate features and labels
X, y = (
    df[
        [
            "Pregnancies",
            "PlasmaGlucose",
            "DiastolicBloodPressure",
            "TricepsThickness",
            "SerumInsulin",
            "BMI",
            "DiabetesPedigree",
            "Age",
        ]
    ].values,
    df["Diabetic"].values,
)

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Initialize and train the classifier
model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
