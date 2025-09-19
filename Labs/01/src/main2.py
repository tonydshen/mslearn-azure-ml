import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.25)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--registered_model_name", type=str, help="Model name")
    parser.add_argument("--experiment_name", type=str, default="credit_defaults_local")
    args = parser.parse_args()

    # Set experiment name
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        mlflow.sklearn.autolog()

        credit_df = pd.read_csv(args.data, header=1, index_col=0)
        train_df, test_df = train_test_split(credit_df, test_size=args.test_train_ratio)

        y_train = train_df.pop("default payment next month")
        X_train = train_df.values
        y_test = test_df.pop("default payment next month")
        X_test = test_df.values

        clf = GradientBoostingClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))

        mlflow.sklearn.log_model(sk_model=clf, registered_model_name=args.registered_model_name, artifact_path=args.registered_model_name)
        # mlflow.sklearn.save_model(sk_model=clf, path=os.path.join(args.registered_model_name, "trained_model"))

if __name__ == "__main__":
    main()
