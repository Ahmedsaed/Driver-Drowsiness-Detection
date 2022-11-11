import argparse
import sys

def get_input_args():
    parser = argparse.ArgumentParser(description='Driver Drowsiness Detection')

    parser.add_argument('-s', '--setup', action='store_true', default = False, help = 'Setup Kaggle API to download data')
    parser.add_argument('-d', '--download_datasets', action='store_true', default = False, help = 'Download Datasets from Kaggle')
    parser.add_argument('-p', '--preprocess_data', action='store_true', default = False, help = 'Apply preprocessing to images')
    parser.add_argument('-t', '--train_model', action='store_true', default = False, help = 'Train model on data')
    parser.add_argument('--train_dir', type=str, default=None, help = 'Path to training directory', required=('--train_model' in sys.argv or '-t' in sys.argv) and '-p' in sys.argv)
    parser.add_argument('--logging_level', type = str, default = 'debug', help = 'Set logging level', choices=['debugging', 'info', 'warning'])

    return parser.parse_args()