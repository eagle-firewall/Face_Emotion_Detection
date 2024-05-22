
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Your directory paths
traindir = "images/train"
testdir = "images/validation"

def create_df(dir):
    img_path = []
    img_label = []
    for i in os.listdir(dir):
        for j in os.listdir(os.path.join(dir, i)):
            img_path.append(os.path.join(dir, i, j))
            img_label.append(i)
        print(i, 'completed')
    return img_path, img_label

train = pd.DataFrame()
train['image'], train['label'] = create_df(traindir)
test = pd.DataFrame()
test['image'], test['label'] = create_df(testdir)

# Convert labels to numeric format
label_encoder = LabelEncoder()
train["label"] = label_encoder.fit_transform(train["label"])
test["label"] = label_encoder.transform(test["label"])

def extract_features(images):
    features = []
    for img_path in tqdm(images):
        img = load_img(img_path, color_mode="grayscale", target_size=(48, 48))
        img_array = img_to_array(img)
        features.append(img_array)
    features = np.array(features)
    return features

train_feature = extract_features(train['image'])
test_feature = extract_features(test['image'])

x_train = train_feature / 255.0
x_test = test_feature / 255.0

# Assuming you have a 'label' column in your 'train' DataFrame
y_train = train["label"].values

# Assuming you have a 'label' column in your 'test' DataFrame
y_test = test["label"].values
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(7, activation='softmax')
])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))

model_json = model.to_json()
with open('emotiondetector2.json', 'w') as j:
    j.write(model_json)

model.save("emotiondetector2.h5")

