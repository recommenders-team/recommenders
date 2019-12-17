import argparse

from azureml.studio.core.logger import module_logger as logger
from reco_utils.dataset.python_splitters import python_stratified_split
from azureml.studio.core.data_frame_schema import DataFrameSchema
from azureml.studio.core.io.data_frame_directory import load_data_frame_from_directory, save_data_frame_to_directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-path',
        help='The input directory.',
    )

    parser.add_argument(
        '--ratio', type=float,
        help='A float parameter.',
    )

    parser.add_argument(
        '--col-user', type=str,
        help='A string parameter.',
    )

    parser.add_argument(
        '--col-item', type=str,
        help='A string parameter.',
    )

    parser.add_argument(
        '--seed', type=int,
        help='An int parameter.',
    )

    parser.add_argument(
        '--output-train',
        help='The output training data directory.',
    )
    parser.add_argument(
        '--output-test',
        help='The output test data directory.',
    )

    args, _ = parser.parse_known_args()

    input_df = load_data_frame_from_directory(args.input_path).data

    #logger.info(f"Hello world from {PACKAGE_NAME} {VERSION}")

    ratio = args.ratio
    col_user = args.col_user
    col_item = args.col_item
    seed = args.seed

    logger.debug(f"Received parameters:")
    logger.debug(f"Ratio:    {ratio}")
    logger.debug(f"User:    {col_user}")
    logger.debug(f"Item:    {col_item}")
    logger.debug(f"Seed:    {seed}")

    logger.debug(f"Input path: {args.input_path}")
    logger.debug(f"Shape of loaded DataFrame: {input_df.shape}")
    logger.debug(f"Cols of DataFrame: {input_df.columns}")

    output_train, output_test = python_stratified_split(input_df, ratio=args.ratio, col_user=args.col_user, col_item=args.col_item, seed=args.seed)

    logger.debug(f"Output path: {args.output_train}")
    logger.debug(f"Output path: {args.output_test}")

    save_data_frame_to_directory(args.output_train, output_train, schema=DataFrameSchema.data_frame_to_dict(output_train))
    save_data_frame_to_directory(args.output_test, output_test, schema=DataFrameSchema.data_frame_to_dict(output_test))

