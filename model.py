from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import numpy as np
import os
import logging

def load_model(load_last=True):
    if load_last and os.path.exists(os.path.join('.', 'Models')) and len(os.listdir(os.path.join(".", "Models"))) > 0:
        logging.info('Loading model from {os.path.join(".", "Models", f"model{len(os.listdir(os.path.join(".", "Models")))}.h5")}')
        model = load_model(os.path.join('.', 'Models', f'model{len(os.listdir(os.path.join(".", "Models")))}.h5'))
    else:
        logging.info('No model found. Creating a new one')
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


            Dense(512, activation='relu'),
            Dropout(0.1),
            Dense(1, activation='sigmoid')
        ])

        model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

    model.summary()

    return model

def train(train_generator, test_generator, load_last=True):
    logging.info('Loading model for training')
    model = load_model(load_last)

    model.fit(train_generator, epochs=10, validation_data=test_generator)

    model.save(os.path.join('.', 'Models', f'model{len(os.listdir(os.path.join(".", "Models")))+1}.h5'))

def evaluate(test_generator):
    logging.info('Loading model for evaluation')
    model = load_model()
    result = model.evaluate(test_generator)
    metrics = dict(zip(model.metrics_names, result))
    print(f"Model Evaluation Score:")
    print(f"Loss: {metrics['loss']}")
    print(f"Accuracy: {metrics['accuracy']}")
