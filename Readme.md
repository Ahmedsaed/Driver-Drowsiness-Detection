# Driver Drowsiness Detection

This project aims to reduce car accidents by detecting if the driver is a wake or not then take a specific action. for our case, we use a recommendation system to play energetic music to help the drive stay awake and focused

**This repo hosts the command line interface and the deployment**

# Installation
## 1. Download these packages

- pip
- kaggle-cli
- python3

## 2. Clone the repo
```bash
git clone https://github.com/Ahmedsaed/Driver-Drowsiness-Detection.git
```

## 3. Update permissions 
```bash
chmod +x run.py 
``` 

## 4. Create .env file
Rename `.env copy` to `.env` and add your Kaggle API Key to the file 


## 5. install python packages
```bash
pip install -r requirements.txt
```

# Usage
Use `./run.py -h` for help

```
usage: run.py [-h] [-s] [-d] [-p] [-t] [--train_dir TRAIN_DIR] [--logging_level {debugging,info,warning}]

Driver Drowsiness Detection

options:
  -h, --help            show this help message and exit
  -s, --setup           Setup Kaggle API to download datasets
  -d, --download_datasets
                        Download Datasets from Kaggle
  -p, --preprocess_data
                        Apply preprocessing to images
  -t, --train_model     Train model on data
  --train_dir TRAIN_DIR
                        Path to training directory
  --logging_level {debugging,info,warning}
                        Set logging level
```

# Setup
Run `./run.py -s -d` to setup kaggle and download the datasets

The `-s` is used to setup directories and files. The `-d` is used to download and extract datasets for training and evaluation

Use `--logging_level` to choose a logging level from `{debugging, info, warning}`

# Training 
Run `./run.py -p -t --train_dir <training dir>`  to preprocess images and train the model

The `-p` is used to trigger image preprocessing which includes adding face landmarks, resizing and image augmentation

The `-t` is used to train the model and `--train_dir` is required when `-p` is given.
if `--train_dir` is not provided the script will load precomputed-preprocessed images from `./Data/landmarks`

# Deploy 
To host the deployment website, Run the following command:
```bash
streamlit run deploy.py
```
Then open the browser and go to `localhost:8501`