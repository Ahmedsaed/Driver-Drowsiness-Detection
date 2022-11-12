import os
import json
import logging
import subprocess
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

api_token = {"username":"ahmedsaed26", "key":os.getenv('KAGGLE_API')}
user_path = os.path.expanduser('~')

def setup_kaggle():
    if not os.path.exists(os.path.join(user_path, '.kaggle')):
        logging.warning('No kaggle configuration found! Creating one')
        os.mkdir(os.path.join(user_path, '.kaggle'))

    with open(os.path.join(user_path, '.kaggle', 'kaggle.json'), 'w') as file:
        logging.info(f'Adding {api_token["username"]}')
        json.dump(api_token, file)


def setup_dirs(categories):
    for cat in categories:
        if not os.path.exists(os.path.join('.', 'Data', 'landmarks', cat)):
            os.makedirs(os.path.join('.', 'Data', 'landmarks', cat))

    if not os.path.exists(os.path.join('.', 'Models')):
        os.makedirs(os.path.join('.', 'Models'))

    if not os.path.exists(os.path.join('.', 'Logs')):
        os.makedirs(os.path.join('.', 'Logs'))

def download_datasets(datasets):
    logging.info('Downloading Datasets')
    for dataset in datasets:
        subprocess.call(['kaggle', 'datasets', 'download', '-d', dataset, '-p', './Data'])

    logging.info('Unzipping Downloaded Datasets')
    for dataset in datasets:
        dataset_name = dataset.split('/')[-1]
        subprocess.call(['unzip', '-q', os.path.join('.', 'Data', dataset_name+'.zip'), '-d', os.path.join('.', 'Data', dataset_name)])

