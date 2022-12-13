# Driver Drowsiness Detection

A computer vision project that aims to reduce car accidents by detecting if the driver is a wake or not then it takes a specific action. for our case, we used a recommendation system to play energetic music to help the driver stay awake and focused.

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

## Setup
Run `./run.py -s -d` to setup kaggle and download the datasets

The `-s` is used to setup directories and files. The `-d` is used to download and extract datasets for training and evaluation

Use `--logging_level` to choose a logging level from `{debugging, info, warning}`

## Training 
Run `./run.py -p -t --train_dir <training dir>`  to preprocess images and train the model

The `-p` is used to trigger image preprocessing which includes adding face landmarks, resizing and image augmentation

The `-t` is used to train the model and `--train_dir` is required when `-p` is given.
if `--train_dir` is not provided the script will load precomputed-preprocessed images from `./Data/landmarks`

## Deploy 
To host the deployment website, Run the following command:
```bash
streamlit run deploy.py
```
Then open the browser and go to `localhost:8501`

# Model Architecture

```python
model = Sequential([
            Conv2D(16, 3, activation='relu', input_shape=(145, 145, 3)),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.1),

            Conv2D(32, 5, activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.1),

            Conv2D(64, 10, activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Dropout(0.1),

            Conv2D(128, 12, activation='relu'),
            BatchNormalization(),

            Flatten(),

            Dense(128, activation='relu'),
            Dropout(0.25),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

```
