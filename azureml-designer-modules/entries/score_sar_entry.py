import argparse
from distutils.util import strtobool
from enum import Enum
from pathlib import Path
import joblib
import time

from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.core.logger import module_logger as logger
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory


class ScoreType(Enum):
    ITEM_RECOMMENDATION = 'Item recommendation'
    RATING_PREDICTION = 'Rating prediction'


class RankingMetric(Enum):
    RATING = 'Rating'
    SIMILARITY = 'Similarity'
    POPULARITY = 'Popularity'


class ItemSet(Enum):
    TRAIN_ONLY = 'Items in training dataset'
    SCORE_ONLY = 'Items in score dataset'


MODEL_NAME = 'sar_model'


def recommend_items(model, data, ranking_metric, top_k, sort_top_k, remove_seen, normalize):
    if ranking_metric == RankingMetric.RATING:
        return model.recommend_k_items(test=data, top_k=top_k, sort_top_k=sort_top_k, remove_seen=remove_seen,
                                       normalize=normalize)
    if ranking_metric == RankingMetric.SIMILARITY:
        return model.get_item_based_topk(items=data, top_k=top_k, sort_top_k=sort_top_k)
    if ranking_metric == RankingMetric.POPULARITY:
        return model.get_popularity_based_topk(top_k=top_k, sort_top_k=sort_top_k)
    raise ValueError(f"Got unexpected ranking metric: {ranking_metric}.")


def predict_ratings(model, data, items_to_predict, remove_seen, normalize):
    if items_to_predict == ItemSet.TRAIN_ONLY:
        return model.score(test=data, remove_seen=remove_seen, normalize=normalize)
    if items_to_predict == ItemSet.SCORE_ONLY:
        return model.predict(test=data)
    raise ValueError(f"Got unexpected 'items to predict': {items_to_predict}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--trained-model', help='The directory contains trained SAR model.')
    parser.add_argument(
        '--dataset-to-score', help='Dataset to score')
    parser.add_argument(
        '--score-type', type=str, help='The type of score which the recommender should output')
    parser.add_argument(
        '--items-to-predict', type=str, help='The set of items to predict for test users')
    parser.add_argument(
        '--normalize', type=str, help='Normalize predictions to scale of original ratings')
    parser.add_argument(
        '--ranking-metric', type=str, help='The metric of ranking used in item recommendation')
    parser.add_argument(
        '--top-k', type=int, help='The number of top items to recommend.')
    parser.add_argument(
        '--sort-top-k', type=str, help='Sort top k results.')
    parser.add_argument(
        '--remove-seen-items', type=str, help='Remove items seen in training from recommendation')
    parser.add_argument(
        '--score-result', help='Ratings or items to output')

    args, _ = parser.parse_known_args()

    logger.info(f"Arguments: {args}")

    with open(Path(args.trained_model) / MODEL_NAME, 'rb') as f:
        sar_model = joblib.load(f)

    dataset_to_score = load_data_frame_from_directory(args.dataset_to_score).data
    logger.debug(f"Shape of loaded DataFrame: {dataset_to_score.shape}")

    sort_top_k = strtobool(args.sort_top_k) if args.sort_top_k else None
    remove_seen_items = strtobool(args.remove_seen_items) if args.remove_seen_items else None
    normalize = strtobool(args.normalize) if args.normalize else None

    start_time = time.time()

    score_type = ScoreType(args.score_type)
    if score_type == ScoreType.ITEM_RECOMMENDATION:
        score_result = recommend_items(model=sar_model,
                                       data=dataset_to_score,
                                       ranking_metric=RankingMetric(args.ranking_metric),
                                       top_k=args.top_k,
                                       sort_top_k=sort_top_k,
                                       remove_seen=args.remove_seen_items,
                                       normalize=normalize)
    elif score_type == ScoreType.RATING_PREDICTION:
        score_result = predict_ratings(model=sar_model,
                                       data=dataset_to_score,
                                       items_to_predict=ItemSet(args.items_to_predict),
                                       remove_seen=remove_seen_items,
                                       normalize=normalize)
    else:
        raise ValueError(f"Got unexpected score type: {score_type}.")

    test_time = time.time() - start_time
    logger.debug("Took {} seconds for score.\n".format(test_time))

    save_data_frame_to_directory(args.score_result, data=score_result,
                                 schema=DataFrameSchema.data_frame_to_dict(score_result))
