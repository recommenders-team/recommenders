import argparse
from distutils.util import strtobool
import time
import joblib
from pathlib import Path

from reco_utils.recommender.sar import SAR

from azureml.studio.core.logger import module_logger as logger
from azureml.studio.core.utils.fileutils import ensure_folder
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory
from azureml.studio.core.io.model_directory import save_model_to_directory


def joblib_dumper(data, file_name=None):
    """Return a dumper to dump a model with pickle."""
    if not file_name:
        file_name = "_data.pkl"

    def model_dumper(save_to):
        full_path = Path(save_to) / file_name
        ensure_folder(Path(save_to))
        with open(full_path, "wb") as fout:
            joblib.dump(data, fout, protocol=4)

        model_spec = {"model_type": "joblib", "file_name": file_name}
        return model_spec

    return model_dumper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", help="The input directory.")
    parser.add_argument("--output-model", help="The output model directory.")
    parser.add_argument("--col-user", type=str, help="A string parameter.")
    parser.add_argument("--col-item", type=str, help="A string parameter.")
    parser.add_argument("--col-rating", type=str, help="A string parameter.")
    parser.add_argument("--col-timestamp", type=str, help="A string parameter.")
    parser.add_argument("--normalize", type=str)
    parser.add_argument("--time-decay", type=str)

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
        timedecay_formula=strtobool(args.time_decay),
    )

    start_time = time.time()

    model.fit(input_df)

    train_time = time.time() - start_time
    print("Took {} seconds for training.".format(train_time))

    save_model_to_directory(
        save_to=args.output_model, model_dumper=joblib_dumper(data=model)
    )
