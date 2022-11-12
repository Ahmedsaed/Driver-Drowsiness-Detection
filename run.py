#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from input_args import get_input_args
from setup_data import setup_kaggle, download_datasets, setup_dirs
from preprocess import process_dataset, setup_training_data, load_landmarks
from model import load_saved_model, train, evaluate

setup_dirs([])

args = get_input_args()
logging.basicConfig(encoding='utf-8', level=logging.DEBUG, filename=os.path.join('.', 'Logs', f'py_log{len(os.listdir("Logs"))}.log'), filemode='w', format='%(asctime)s %(levelname)s %(message)s')

datasets = ['rakibuleceruet/drowsiness-prediction-dataset', 'adinishad/prediction-images']
categories = ["Fatigue Subjects", "Active Subjects"]

if args.setup:
    setup_kaggle()

if args.download_datasets:
    download_datasets(datasets)

if args.train_model:
    if args.preprocess_data:
        processed_images = process_dataset(dir_faces=args.train_dir, categories=categories)
    else:
        processed_images = load_landmarks(categories)

    train_generator, test_generator = setup_training_data(processed_images)

    logging.info('Loading model for training')
    model = load_saved_model(load_last=True)

    train(model, train_generator, test_generator)

    evaluate(model, test_generator)

