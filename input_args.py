import argparse


def get_input_args():
    parser = argparse.ArgumentParser(description='Driver Drowsiness Detection')

    parser.add_argument('--logging_level', type = str, default = 'debug', help = 'Set logging level', choices=['debugging', 'info', 'warning'])
    parser.add_argument('-s', '--setup', action='store_true', default = False, help = 'Setup Kaggle API and download data')
    parser.add_argument('-d', '--download_datasets', action='store_true', default = False, help = 'Download Datasets from Kaggle')

    return parser.parse_args()