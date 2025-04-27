import os
import numpy as np
import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import spatial, stats

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("TensorFlow Version:", tf.__version__)

# Corrected Paths
dir_1 = r'C:\Python Projects\Poverty Prediction Code\Code\train\class_1'
dir_2 = r'C:\Python Projects\Poverty Prediction Code\Code\train\class_2'
dir_3 = r'C:\Python Projects\Poverty Prediction Code\Code\train\class_3'

# Load image file names
image_file_1 = os.listdir(dir_1)
image_file_2 = os.listdir(dir_2)
image_file_3 = os.listdir(dir_3)

# Ensure images exist
if not image_file_1 or not image_file_2 or not image_file_3:
    raise FileNotFoundError("One or more directories are empty!")

# Synthetic Data Generation
def generate_synthetic_data():
    X = np.random.rand(1000, 10, 1)
    y = np.random.randint(0, 2, size=(1000, 1))
    return X, y

X, y = generate_synthetic_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(-1, 10, 1)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(-1, 10, 1)

# RNN Model
model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(10, 1)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Model Checkpoint
checkpoint_path = "model.hdf5"
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min')

# Train Model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, callbacks=[checkpoint])

# Load Best Model
best_model = tf.keras.models.load_model(checkpoint_path)

# Load Pretrained VGG16 Model
basemodel = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Extract Features
def get_feature_vector(img):
    img_resized = cv2.resize(img, (224, 224))
    img_preprocessed = preprocess_input(img_resized.reshape(1, 224, 224, 3))
    feature_vector = basemodel.predict(img_preprocessed)
    return feature_vector

# Similarity Calculation
def calculate_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)

# Select Random Images
low_file = os.path.join(dir_1, image_file_1[0])
med_file = os.path.join(dir_2, image_file_2[0])
high_file = os.path.join(dir_3, image_file_3[0])

img1 = mpimg.imread(low_file)
img2 = mpimg.imread(med_file)
img3 = mpimg.imread(high_file)

f1 = get_feature_vector(img1)
f2 = get_feature_vector(img2)
f3 = get_feature_vector(img3)

print("Similarity between low and med:", calculate_similarity(f1, f2))
print("Similarity between med and high:", calculate_similarity(f2, f3))
print("Similarity between low and high:", calculate_similarity(f1, f3))

# Bootstrapping for Similarity Statistics
def bootstrap_stat(size=10):
    sim_results = {
        "sim11": [], "sim22": [], "sim33": [],
        "sim12": [], "sim23": [], "sim13": []
    }

    for _ in range(size):
        # Randomly select image indices
        bs1 = np.random.choice(len(image_file_1), size=2, replace=False)
        bs2 = np.random.choice(len(image_file_2), size=2, replace=False)
        bs3 = np.random.choice(len(image_file_3), size=2, replace=False)

        # Load Images
        img11 = mpimg.imread(os.path.join(dir_1, image_file_1[bs1[0]]))
        img12 = mpimg.imread(os.path.join(dir_1, image_file_1[bs1[1]]))

        img21 = mpimg.imread(os.path.join(dir_2, image_file_2[bs2[0]]))
        img22 = mpimg.imread(os.path.join(dir_2, image_file_2[bs2[1]]))

        img31 = mpimg.imread(os.path.join(dir_3, image_file_3[bs3[0]]))
        img32 = mpimg.imread(os.path.join(dir_3, image_file_3[bs3[1]]))

        # Compute Features
        f11, f12 = get_feature_vector(img11), get_feature_vector(img12)
        f21, f22 = get_feature_vector(img21), get_feature_vector(img22)
        f31, f32 = get_feature_vector(img31), get_feature_vector(img32)

        # Compute Similarities
        sim_results["sim11"].append(calculate_similarity(f11, f12))
        sim_results["sim22"].append(calculate_similarity(f21, f22))
        sim_results["sim33"].append(calculate_similarity(f31, f32))
        sim_results["sim12"].append(calculate_similarity(f11, f21))
        sim_results["sim23"].append(calculate_similarity(f21, f31))
        sim_results["sim13"].append(calculate_similarity(f11, f31))

    return pd.DataFrame(sim_results)

# Perform Bootstrapping
df_sim = bootstrap_stat(size=1000)
print(df_sim.describe())

# Plot Histograms
ax = df_sim.sim11.hist(bins=100, alpha=0.5, label='sim11')
df_sim.sim22.hist(bins=100, alpha=0.5, ax=ax, label='sim22')
df_sim.sim33.hist(bins=100, alpha=0.5, ax=ax, label='sim33')
plt.legend()
plt.show()

# Perform T-Test
rvs1, rvs2, rvs3 = df_sim.sim11.values, df_sim.sim22.values, df_sim.sim33.values
t_stat, p_val = stats.ttest_ind(rvs1, rvs2, equal_var=False)
print(f"T-test between sim11 and sim22: t={t_stat:.4f}, p={p_val:.4f}")
