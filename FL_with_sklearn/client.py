import warnings
import flwr as fl
import numpy as np
import argparse
import wandb 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score

import utils
wandb.init(project="FL-trying", entity="circoval1001")
if __name__ == "__main__":
    file_paths = ['../nsl-kdd/KDDTrain+.txt', '../nsl-kdd/KDDTest+.txt']
    
    (X_train, y_train), (X_test, y_test) = utils.preprocessing(
        file_path_train=file_paths[0], file_path_test=file_paths[1])
    
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]
    
    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class Client(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_params(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            return utils.get_model_params(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            wandb.log({"train_loss": loss})
            wandb.log({"accuracy": accuracy})
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8081", client=Client())
