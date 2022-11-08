#!/usr/bin/env python

import logging
from input_args import get_input_args
from setup_data import setup_kaggle, download_datasets

args = get_input_args()
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
datasets = ['dheerajperumandla/drowsiness-dataset', 'adinishad/prediction-images']

if args.setup:
    setup_kaggle()

if args.download_datasets:
    download_datasets(datasets)

