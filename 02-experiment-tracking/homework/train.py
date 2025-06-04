# mlflow ui --backend-store-uri sqlite:///mlflow.db

import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import mlflow


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('my-first-experiment')

    with mlflow.start_run():

        mlflow.set_tag('developer', 'OcheAI')
        mlflow.log_param('train-data-path', './output/train.pkl' )
        mlflow.log_param('test-data-path', './output/test.pkl')
        mlflow.log_param('val-data-path', './output/val.pkl')

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        max_depth=10

        mlflow.log_param('max_depth', max_depth)

        rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric('rmse', rmse)


if __name__ == '__main__':
    run_train()