import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        default="./configs/vt_mapillary.yaml",
        type=str,
        help="Config file location",
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Set to execute training and not validation",
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=4,
        type=int,
        help="Batch size to use in training and validation",
    )
    args = parser.parse_args()
    return args
