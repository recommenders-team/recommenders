# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
This Python script completes post test tasks such as downloading logs.
"""

import argparse
import mlflow
import logging
import pathlib

from aml_utils import get_client, correct_resource_name


def parse_args():
    """
    Parse command line arguments.
    """

    parser = argparse.ArgumentParser(description="Process some inputs")

    parser.add_argument(
        "--rg", action="store",
        default="recommender",
        help="Azure Resource Group",
    )
    parser.add_argument(
        "--ws", action="store",
        default="RecoWS",
        help="AzureML workspace name",
    )
    parser.add_argument(
        "--subid",
        action="store",
        default="123456",
        help="Azure Subscription ID",
    )
    parser.add_argument(
        "--expname",
        action="store",
        default="persistentAzureML",
        help="Experiment name on AzureML",
    )
    parser.add_argument(
        "--log-dir",
        action="store",
        default="test_logs",
        help="Test logs will be downloaded to this path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger("post_pytest.py")
    args = parse_args()

    logger.info(f"Setting up workspace {args.ws}")
    client = get_client(
        subscription_id=args.subid,
        resource_group=args.rg,
        workspace_name=args.ws,
    )

    # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-configure-tracking?view=azureml-api-2&tabs=python%2Cmlflow#configure-mlflow-tracking-uri
    logger.info(f"Configuring mlflow")
    mlflow.set_tracking_uri(
        client.workspaces.get(client.workspace_name).mlflow_tracking_uri
    )

    # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments-mlflow?view=azureml-api-2
    logger.info(f"Searching runs")
    experiment_name = correct_resource_name(args.expname)
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        max_results=1,
        output_format="list",
    )
    if runs:
        run = runs[0]

        # See https://www.mlflow.org/docs/latest/python_api/mlflow.artifacts.html#mlflow.artifacts.download_artifacts
        # For more details on logs, see
        # * https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics?view=azureml-api-2&tabs=interactive#view-and-download-diagnostic-logs
        # * https://azure.github.io/azureml-cheatsheets/docs/cheatsheets/python/v1/debugging/
        logger.info(f"Downloading AzureML logs")
        mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            dst_path=args.log_dir,
        )
        log_path = next(
            (path for path in pathlib.Path(args.log_dir).rglob("std_log.txt")),
            None
        )
        if log_path is not None:
            with open(log_path, "r") as file:
                print(f"\nDumping logs in {log_path}")
                print("=====================================")
                print(file.read())
