#!/usr/bin/env python

import logging
from input_args import get_input_args
from setup_data import setup_kaggle, download_datasets
from preprocess import preprocess_images
from model import train

args = get_input_args()
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

datasets = ['dheerajperumandla/drowsiness-dataset', 'adinishad/prediction-images']
categories = ["drowse", "not_drowse"]

if args.setup:
    setup_kaggle()

if args.download_datasets:
    download_datasets(datasets)

if args.train_dir:
    processed_images = ...
    if args.preprocess_data:
        processed_images = preprocess_images(args.train_dir)

    train(processed_images, args.train_dir)