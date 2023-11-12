import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import wandb 
import argparse

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    file_paths = ['../nsl-kdd/KDDTrain+.txt', '../nsl-kdd/KDDTest+.txt']

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.preprocessing(file_path_train=file_paths[0], file_path_test=file_paths[1])

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower-Server")
    parser.add_argument("--strategy", type=str, default= 'fedavg')
    parser.add_argument("--fr_rate", type=float, default=0.0)
    parser.add_argument("--fr_val_rate", type=float, default=0.0)
    parser.add_argument("--min_client", type=int, default=2)
    parser.add_argument("--min_ac", type=int, default=2)
    args = parser.parse_args()

    model = LogisticRegression()
    utils.set_initial_params(model)
    if args.strategy == 'fedavg' : 
        strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
        )
    elif args.strategy == 'fedmedian' : 
        strategy = fl.server.strategy.FedMedian(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
        )
    elif args.strategy == 'fedavgQ' : 
        strategy = fl.server.strategy.QFedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
        )
    elif args.strategy == 'fedfaulttol' : 
        strategy = fl.server.strategy.FaultTolerantFedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
        )
    elif args.strategy == 'fedtrim' : 
        strategy = fl.server.strategy.FedTrimmedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
        )
    elif args.strategy == 'FedXgbNn' : 
        strategy = fl.server.strategy.FedXgbNnAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
        )
   
    wandb.init(project="FL-trying", entity="circoval1001")
    wandb.config.update(args)
    fl.server.start_server(
        server_address="127.0.0.1:8081",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20),
    )
