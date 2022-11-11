#!/usr/bin/env python

import logging
from input_args import get_input_args
from setup_data import setup_kaggle, download_datasets
from preprocess import process_dataset
from model import train

args = get_input_args()
logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

datasets = ['rakibuleceruet/drowsiness-prediction-dataset', 'adinishad/prediction-images']
categories = ["Fatigue Subjects", "Active Subjects"]

if args.setup:
    setup_kaggle()

if args.download_datasets:
    download_datasets(datasets)

if args.train_dir:
    processed_images = ...
    if args.preprocess_data:
        processed_images = process_dataset(dir_faces=args.train_dir, categories=categories)

    train(processed_images)