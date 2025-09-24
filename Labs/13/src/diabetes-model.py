# import libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# load the diabetes dataset
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Path to input data")
parser.add_argument("--test_size", type=float, default=0.30, help="Proportion of the dataset to include in the test split")
parser.add_argument("--reg", type=float, default=0.01, help="Regularization rate parameter")
parser.add_argument("--registered_model_name", type=str, help="Model name")

args = parser.parse_args()
args

print("Loading Data...")
diabetes = pd.read_csv(args.data)

# Start an MLflow run
mlflow.start_run()
mlflow.sklearn.autolog()

# separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

# set regularization hyperparameter
reg = args.reg

# train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg, 'test size of', args.test_size)
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))

# mlflow.sklearn.log_model(sk_model=model, registered_model_name=args.registered_model_name, artifact_path=args.registered_model_name)
mlflow.sklearn.log_model(sk_model=model, registered_model_name=args.registered_model_name, name=args.registered_model_name)

mlflow.end_run()
