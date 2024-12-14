# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# Import necessary modules first
from tensorflow.keras.utils import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings
from tqdm.notebook import tqdm
warnings.filterwarnings('ignore')
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

!unzip '/content/drive/MyDrive/archive.zip'

BASE_DIR = '/content/UTKFace'
age_labels = []
gender_labels = []
image_paths = []

image_filenames = os.listdir(BASE_DIR)
random.shuffle(image_filenames)

for image in tqdm(image_filenames):
  image_path = os.path.join(BASE_DIR, image)
  img_components = image.split('_')
  age_label = int(img_components[0])
  gender_label = int(img_components[1])

  # Append the image_path, age_label, and gender_label
  image_paths.append(image_path)
  age_labels.append(age_label)
  gender_labels.append(gender_label)

print(f"Number of age_labels: {len(age_labels)}")
print(f"Number of gender_labels: {len(gender_labels)}")
print(f"Number of image_paths: {len(image_paths)}")

gender_mapping = {0: 'Male', 1: 'Female'}

import pandas as pd
df = pd.DataFrame()
df['image_path'] = image_paths
df['age'] = age_labels
df['gender'] = gender_labels
df.head()

"""#Exploratory Data Analysis"""

from PIL import Image

rand_index = random.randint(0, len(image_paths))
age = df['age'][rand_index]
gender = df['gender'][rand_index]
image_path = df['image_path'][rand_index]

img = load_img(image_path)
plt.imshow(img)
plt.title(f"Age: {age} | Gender: {gender_mapping[gender]}")

from PIL import Image

rand_index = random.randint(0, len(image_paths))
age = df['age'][rand_index]
gender = df['gender'][rand_index]
image_path = df['image_path'][rand_index]

img = load_img(image_path)
plt.imshow(img)
plt.title(f"Age: {age} | Gender: {gender_mapping[gender]}")

# Age distribution
sns.distplot(df['age'])

"""The distribution roughly follows a normal distribution that is slightly skewed to the right with a median of around 27 years. The range is from 0 to 120 years. There are some outliers at the higher end of the distribution."""

plt.figure(figsize=(20, 20))
samples = df.iloc[0:16]

for index, sample, age, gender in samples.itertuples():
  plt.subplot(4, 4, index+1)
  img = load_img(sample)
  plt.imshow(img)
  plt.title(f"Age: {age} | Gender: {gender_mapping[gender]}")

"""#Feature Extraction"""

import os

for image in df['image_path']:
    if not os.path.exists(image):
        print(f"Image not found: {image}")

def extract_image_features(images):
    features = list()

    for idx, image in enumerate(tqdm(images)):
        try:
            img = load_img(image, color_mode="grayscale")
            img = img.resize((128, 128), Image.ANTIALIAS)
            img = np.array(img)
            features.append(img)

            if idx % 100 == 0:  # Print progress every 100 images
                print(f"Processed {idx + 1}/{len(images)} images")

        except Exception as e:
            print(f"Error processing image {image}: {e}")

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)

    return features

X = extract_image_features(df['image_path'])

print(X.shape)

X = X/255.0

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

input_shape = (128, 128, 1)

inputs = Input((input_shape))
conv_1 = Conv2D(32, kernel_size=(3,3), activation='relu')(inputs)
max_1 = MaxPooling2D(pool_size=(2,2))(conv_1)
conv_2 = Conv2D(64, kernel_size=(3,3), activation='relu')(max_1)
max_2 = MaxPooling2D(pool_size=(2,2))(conv_2)
conv_3 = Conv2D(128, kernel_size=(3,3), activation='relu')(max_2)
max_3 = MaxPooling2D(pool_size=(2,2))(conv_3)
conv_4 = Conv2D(256, kernel_size=(3,3), activation='relu')(max_3)
max_4 = MaxPooling2D(pool_size=(2,2))(conv_4)

flatten = Flatten()(max_4)

#fully connected layers
dense_1 = Dense(256, activation='relu')(flatten)
dense_2 = Dense(256, activation='relu')(flatten)

dropout_1 = Dropout(0.3)(dense_1)
dropout_2 = Dropout(0.3)(dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_output')(dropout_1)
output_2 = Dense(1, activation='linear', name='age_output')(dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy','mae'],
              optimizer='adam', metrics=['accuracy'])

from sklearn.model_selection import train_test_split
import numpy as np

# Assuming X, y_gender, and y_age are already defined
# X should be your images, y_gender the gender labels, and y_age the age labels.

# Example:
# X = np.array([...])         # Your image data
# y_gender = np.array([...])  # Your gender labels
# y_age = np.array([...])     # Your age labels

# Split the data into training and validation sets
X_train, X_val, y_gender_train, y_gender_val, y_age_train, y_age_val = train_test_split(
    X, y_gender, y_age, test_size=0.2, random_state=42
)

model.compile(loss=['binary_crossentropy', 'mae'],
              optimizer='adam',
              metrics=[['accuracy'], ['mae']])

#plot the model
from tensorflow.keras.utils import plot_model
plot_model(model)

history = model.fit(x=X_train, y=[y_gender_train, y_age_train],
                    validation_data=(X_val, [y_gender_val, y_age_val]),
                    batch_size=32, epochs=50)

#plot result for gender
acc = history.history['gender_output_accuracy']
val_acc = history.history['val_gender_output_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

#plot result for loss
loss = history.history['age_output_mae']
val_loss = history.history['val_age_output_mae']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

#plot results for age
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

"""#Prediction with Test Data"""

def get_image_features(image):
    img = load_img(image, color_mode="grayscale")
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img)
    img = img.reshape(1, 128, 128, 1)
    img = img/255.0
    return img

"""**WRONG**


"""

img_to_test = '/content/Screenshot (193).png'
features = get_image_features(img_to_test)
pred = model.predict(features)
gender = gender_mapping[round(pred[0][0][0])]
age = round(pred[1][0][0])

plt.title(f'Predicted Age: {age} Predicted Gender: {gender}')
plt.axis('off')
plt.imshow(np.array(load_img(img_to_test)))

"""**CORRECT**"""

img_to_test = '/content/Screenshot 2024-08-21 191747.png'
features = get_image_features(img_to_test)
pred = model.predict(features)
gender = gender_mapping[round(pred[0][0][0])]
age = round(pred[1][0][0])

plt.title(f'Predicted Age: {age} Predicted Gender: {gender}')
plt.axis('off')
plt.imshow(np.array(load_img(img_to_test)))

"""**WRONG**"""

img_to_test = '/content/Screenshot 2024-08-21 191818.png'
features = get_image_features(img_to_test)
pred = model.predict(features)
gender = gender_mapping[round(pred[0][0][0])]
age = round(pred[1][0][0])

plt.title(f'Predicted Age: {age} Predicted Gender: {gender}')
plt.axis('off')
plt.imshow(np.array(load_img(img_to_test)))











# Plot Gender Accuracy
plt.plot(history.history['gender_output_accuracy'], label='Training Gender Accuracy')
plt.plot(history.history['val_gender_output_accuracy'], label='Validation Gender Accuracy')
plt.title('Gender Accuracy')
plt.legend()
plt.show()

# Plot Age MAE
plt.plot(history.history['age_output_mae'], label='Training Age MAE')
plt.plot(history.history['val_age_output_mae'], label='Validation Age MAE')
plt.title('Age MAE')
plt.legend()
plt.show()

"""**CORRECT**"""

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def preprocess_image(image_path, target_size=(128, 128)):
    # Load the image
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img = img_to_array(img)  # Convert to array

    # Normalize pixel values
    img = img / 255.0

    # Expand dimensions to match the model's input shape
    img = np.expand_dims(img, axis=0)

    return img

image_path = '/content/Screenshot 2024-08-21 184702.png'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the trained model
gender_pred, age_pred = model.predict(preprocessed_image)

# Gender Prediction (0 for female, 1 for male, as the final layer is 'sigmoid')
gender = "Male" if gender_pred[0] > 0.5 else "Female"

# Age Prediction (this is a regression output, so it's a numerical value)
# Extract the scalar value from the array
age = age_pred[0][0]

print(f"Predicted Gender: {gender}")
print(f"Predicted Age: {age:.1f} years")

import matplotlib.pyplot as plt
from PIL import Image

# Display the image
img = Image.open(image_path)
plt.imshow(img)
plt.title(f"Predicted Gender: {gender}, Predicted Age: {age:.1f} years")
plt.axis('off')
plt.show()

"""**CORRECT**"""

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def preprocess_image(image_path, target_size=(128, 128)):
    # Load the image
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img = img_to_array(img)  # Convert to array

    # Normalize pixel values
    img = img / 255.0

    # Expand dimensions to match the model's input shape
    img = np.expand_dims(img, axis=0)

    return img

image_path = '/content/Screenshot 2024-08-21 184842.png'  # Replace with the path to your image
preprocessed_image = preprocess_image(image_path)

# Predict using the trained model
gender_pred, age_pred = model.predict(preprocessed_image)

# Gender Prediction (0 for female, 1 for male, as the final layer is 'sigmoid')
gender = "Male" if gender_pred[0] > 0.5 else "Female"

# Age Prediction (this is a regression output, so it's a numerical value)
# Extract the scalar value from the array
age = age_pred[0][0]

print(f"Predicted Gender: {gender}")
print(f"Predicted Age: {age:.1f} years")

import matplotlib.pyplot as plt
from PIL import Image

# Display the image
img = Image.open(image_path)
plt.imshow(img)
plt.title(f"Predicted Gender: {gender}, Predicted Age: {age:.1f} years")
plt.axis('off')
plt.show()