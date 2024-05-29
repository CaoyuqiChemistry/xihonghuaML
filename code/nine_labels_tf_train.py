# -*- coding: utf-8 -*-
"""
Created on 5/14/2024 10:36:49

@FileName: nine_labels_tf_train.py
@Author: Cao Yuqi
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

class ZCConvClass(tf.keras.Model):
    def __init__(self):
        super(ZCConvClass, self).__init__()
        self.vgg = models.Sequential([
                        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
                        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                        tf.keras.layers.MaxPooling2D((64, 64), strides=(64, 64)),
                        tf.keras.layers.Flatten(),
                        ])
        self.classifier = models.Sequential([
                            layers.Dense(units = 512, activation='tanh'),
                            layers.Dropout(rate = 0.5),
                            layers.Dense(units = 256, activation='tanh'),
                            layers.Dense(units = 9, activation='softmax') # Totally two classes, so units = 2
                            ])

    def call(self, inputs, training=False):
          inputs = self.vgg(inputs)
          outputs = self.classifier(inputs)
          return outputs

AUTOTUNE = tf.data.experimental.AUTOTUNE

def dataset_generate(data_dir,validation_split = 0.3,seed = 123,image_size = [224, 224],batch_size = 1):
    ds_train_ = image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        color_mode="rgb",
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
    )

    ds_valid_ = image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        color_mode="rgb",
        interpolation='nearest',
        batch_size=batch_size,
        shuffle=True,
    )

    ds_train = (
        ds_train_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )

    ds_valid = (
        ds_valid_
        .map(convert_to_float)
        .cache()
        .prefetch(buffer_size=AUTOTUNE)
    )
    return ds_train,ds_valid,ds_train_.class_names

def plot_confusion_matrix(y_true, y_pred, labels):
    num = len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    num_classes = len(labels)
    cm_normalized = np.zeros_like(cm, dtype=np.float64)

    for i in range(num_classes):
        true_count = np.sum(y_true == i)
        if true_count > 0:
            cm_normalized[i, :] = cm[i, :] / true_count

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix, n={num}')
    plt.show()

if __name__ == '__main__':
    data_dir = 'D:\\xihonghuaML\\trainingSet'
    lr = 0.001
    ds_train,ds_valid,labels = dataset_generate(data_dir,batch_size=1)

    model = ZCConvClass()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(ds_train, validation_data=ds_valid, epochs=25)

    model.save('D:\\xihonghuaML\\CNNmodel\\model20240515-1.h5')

    history_frame = pd.DataFrame(history.history)

    history_frame.loc[:, ['loss', 'val_loss']].plot()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.show()

    y_true = extract_data(ds_valid, iter=1)

    predictions = model.predict(ds_valid)
    y_pred = np.argmax(predictions, axis=1)

    plot_confusion_matrix(y_true, y_pred, labels)


