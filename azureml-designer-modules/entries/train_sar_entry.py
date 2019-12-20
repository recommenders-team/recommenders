import argparse
from distutils.util import strtobool
import time
import joblib
from pathlib import Path

from reco_utils.recommender.sar import SAR

from azureml.studio.core.logger import module_logger as logger
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory


MODEL_NAME = 'sar_model'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-path', help='The input directory.')
    parser.add_argument('--output-model', help='The output model directory.')
    parser.add_argument('--col-user', type=str, help='A string parameter.')
    parser.add_argument('--col-item', type=str, help='A string parameter.')
    parser.add_argument('--col-rating', type=str, help='A string parameter.')
    parser.add_argument('--col-timestamp', type=str, help='A string parameter.')
    parser.add_argument('--normalize', type=str)
    parser.add_argument('--time-decay', type=str)

    args, _ = parser.parse_known_args()

    input_df = load_data_frame_from_directory(args.input_path).data
    input_df[args.col_rating] = input_df[args.col_rating].astype(float)

    logger.debug(f"Shape of loaded DataFrame: {input_df.shape}")
    logger.debug(f"Cols of DataFrame: {input_df.columns}")

    model = SAR(
        col_user=args.col_user,
        col_item=args.col_item,
        col_rating=args.col_rating,
        col_timestamp=args.col_timestamp,
        normalize=strtobool(args.normalize),
        timedecay_formula=strtobool(args.time_decay)
    )

    start_time = time.time()

    model.fit(input_df)

    train_time = time.time() - start_time
    print("Took {} seconds for training.".format(train_time))

    model_dir = Path(args.output_model)
    if not model_dir.exists():
        model_dir.mkdir()
    with open(model_dir / MODEL_NAME, 'wb') as f:
        joblib.dump(model, f, protocol=4)
