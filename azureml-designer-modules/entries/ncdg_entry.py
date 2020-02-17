import argparse
import pandas as pd

from azureml.core import Run
from azureml.studio.core.logger import module_logger as logger
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.core.io.data_frame_directory import (
    load_data_frame_from_directory,
    save_data_frame_to_directory,
)
from reco_utils.evaluation.python_evaluation import ndcg_at_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rating-true", help="True DataFrame.")
    parser.add_argument("--rating-pred", help="Predicted DataFrame.")
    parser.add_argument(
        "--col-user", type=str, help="A string parameter with column name for user."
    )
    parser.add_argument(
        "--col-item", type=str, help="A string parameter with column name for item."
    )
    parser.add_argument(
        "--col-rating", type=str, help="A string parameter with column name for rating."
    )
    parser.add_argument(
        "--col-prediction",
        type=str,
        help="A string parameter with column name for prediction.",
    )
    parser.add_argument(
        "--relevancy-method",
        type=str,
        help="method for determining relevancy ['top_k', 'by_threshold'].",
    )
    parser.add_argument("--k", type=int, help="number of top k items per user.")
    parser.add_argument(
        "--threshold", type=float, help="threshold of top items per user."
    )
    parser.add_argument("--score-result", help="Result of the computation.")

    args, _ = parser.parse_known_args()

    rating_true = load_data_frame_from_directory(args.rating_true).data
    rating_pred = load_data_frame_from_directory(args.rating_pred).data

    col_user = args.col_user
    col_item = args.col_item
    col_rating = args.col_rating
    col_prediction = args.col_prediction
    relevancy_method = args.relevancy_method
    k = args.k
    threshold = args.threshold

    logger.debug(f"Received parameters:")
    logger.debug(f"User:       {col_user}")
    logger.debug(f"Item:       {col_item}")
    logger.debug(f"Rating:     {col_rating}")
    logger.debug(f"Prediction: {col_prediction}")
    logger.debug(f"Relevancy:  {relevancy_method}")
    logger.debug(f"K:          {k}")
    logger.debug(f"Threshold:  {threshold}")

    logger.debug(f"Rating True path: {args.rating_true}")
    logger.debug(f"Shape of loaded DataFrame: {rating_true.shape}")
    logger.debug(f"Rating Pred path: {args.rating_pred}")
    logger.debug(f"Shape of loaded DataFrame: {rating_pred.shape}")

    eval_ndcg = ndcg_at_k(
        rating_true,
        rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    score_result = pd.DataFrame({"ndcg_at_k": [eval_ndcg]})
    logger.debug(f"Score: {args.score_result}")

    # Log to AzureML dashboard
    run = Run.get_context()
    run.parent.log("nDCG at {}".format(k), eval_ndcg)
    run.log("ndcg_at_{}".format(k), eval_ndcg)

    save_data_frame_to_directory(
        args.score_result,
        score_result,
        schema=DataFrameSchema.data_frame_to_dict(score_result),
    )

