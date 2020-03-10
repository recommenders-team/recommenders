import argparse
from distutils.util import strtobool
from enum import Enum
from pathlib import Path
import joblib

from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.core.logger import module_logger as logger
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory
from azureml.studio.core.io.model_directory import load_model_from_directory


class ScoreType(Enum):
    ITEM_RECOMMENDATION = 'Item recommendation'
    RATING_PREDICTION = 'Rating prediction'


class RankingMetric(Enum):
    RATING = 'Rating'
    SIMILARITY = 'Similarity'
    POPULARITY = 'Popularity'


class ItemSet(Enum):
    TRAIN_ONLY = 'Items in training set'
    SCORE_ONLY = 'Items in score set'


def joblib_loader(load_from_dir, model_spec):
    file_name = model_spec['file_name']
    with open(Path(load_from_dir) / file_name, 'rb') as fin:
        return joblib.load(fin)


class ScoreSARModule:
    def __init__(self, model, input_data):
        self._model = model
        self._input_data = input_data

    @property
    def model(self):
        return self._model

    @property
    def input_data(self):
        return self._input_data

    def recommend_items(self, ranking_metric, top_k, sort_top_k, remove_seen, normalize):
        if ranking_metric == RankingMetric.RATING:
            return self.model.recommend_k_items(test=self.input_data, top_k=top_k, sort_top_k=sort_top_k,
                                                remove_seen=remove_seen, normalize=normalize)
        if ranking_metric == RankingMetric.SIMILARITY:
            return self.model.get_item_based_topk(items=self.input_data, top_k=top_k, sort_top_k=sort_top_k)
        if ranking_metric == RankingMetric.POPULARITY:
            return self.model.get_popularity_based_topk(top_k=top_k, sort_top_k=sort_top_k)
        raise ValueError(f"Got unexpected ranking metric: {ranking_metric}.")

    def predict_ratings(self, items_to_predict, normalize):
        if items_to_predict == ItemSet.TRAIN_ONLY:
            return self.model.predict_training_items(test=self.input_data, normalize=normalize)
        if items_to_predict == ItemSet.SCORE_ONLY:
            return self.model.predict(test=self.input_data, normalize=normalize)
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
    sort_top_k = strtobool(args.sort_top_k) if args.sort_top_k else None
    remove_seen_items = strtobool(args.remove_seen_items) if args.remove_seen_items else None
    normalize = strtobool(args.normalize) if args.normalize else None

    sar_model = load_model_from_directory(args.trained_model, model_loader=joblib_loader).data
    dataset_to_score = load_data_frame_from_directory(args.dataset_to_score).data
    logger.debug(f"Shape of loaded DataFrame: {dataset_to_score.shape}")

    score_sar_module = ScoreSARModule(model=sar_model, input_data=dataset_to_score)

    score_type = ScoreType(args.score_type)
    if score_type == ScoreType.ITEM_RECOMMENDATION:
        score_result = score_sar_module.recommend_items(ranking_metric=RankingMetric(args.ranking_metric),
                                                        top_k=args.top_k, sort_top_k=sort_top_k,
                                                        remove_seen=args.remove_seen_items, normalize=normalize)
    elif score_type == ScoreType.RATING_PREDICTION:
        score_result = score_sar_module.predict_ratings(items_to_predict=ItemSet(args.items_to_predict),
                                                        normalize=normalize)
    else:
        raise ValueError(f"Got unexpected score type: {score_type}.")

    save_data_frame_to_directory(args.score_result, data=score_result,
                                 schema=DataFrameSchema.data_frame_to_dict(score_result))
