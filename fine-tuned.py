import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow_hub as hub
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Function to preprocess images (customize based on your need)
def preprocess_image(img_path, target_size=(224, 224)):
    img = Image.open(img_path).resize(target_size)
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    return img

# Load pre-trained BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Load pre-trained image model (e.g., ResNet)
image_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", trainable=False)

# Define model inputs
text_input = Input(shape=(512,), dtype='int32', name='text_input')
mask_input = Input(shape=(512,), dtype='int32', name='mask_input')
image_input = Input(shape=(224, 224, 3), name='image_input')

# Text features
text_features = bert_model(text_input, attention_mask=mask_input)[1]

# Image features
image_features = image_model(image_input)

# Combine features and add classification layers
combined_features = Concatenate()([text_features, image_features])
combined_features = Dense(1024, activation='relu')(combined_features)
output = Dense(num_classes, activation='softmax')(combined_features)  # num_classes based on your dataset

# Build the model
model = Model(inputs=[text_input, mask_input, image_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Example data preparation (adjust based on your dataset)
questions = [...]  # list of question texts
answers = [...]    # list of answer texts
images = [...]     # list of image file paths
labels = [...]     # list of labels (integers)

# Tokenize text data
max_length = 512
text_data = [bert_tokenizer.encode_plus(q + " " + a, max_length=max_length, pad_to_max_length=True, return_tensors='tf') for q, a in zip(questions, answers)]
input_ids = np.array([t['input_ids'][0] for t in text_data])
attention_masks = np.array([t['attention_mask'][0] for t in text_data])

# Preprocess images
image_data = np.array([preprocess_image(img_path) for img_path in images])

# Convert labels to one-hot encoding
label_data = to_categorical(labels, num_classes=num_classes)

# Split the data into training and testing sets
train_inputs, test_inputs, train_masks, test_masks, train_images, test_images, train_labels, test_labels = train_test_split(input_ids, attention_masks, image_data, label_data, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_masks, train_images, train_labels)).shuffle(len(train_labels)).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_masks, test_images, test_labels)).batch(32)

# Train the model
history = model.fit(train_dataset, epochs=5)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Predict and analyze results (optional)
predictions = model.predict(test_dataset)
predicted_labels = np.argmax(predictions, axis=1)
test_actual_labels = np.argmax(test_labels, axis=1)
print(classification_report(test_actual_labels, predicted_labels))
print(confusion_matrix(test_actual_labels, predicted_labels))
