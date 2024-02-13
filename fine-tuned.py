import re
import numpy as np
from PIL import Image
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
import tensorflow_hub as hub

def load_and_preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().split('\n\n')  # Split data into blocks

    questions, answers, images, labels = [], [], [], []
    for block in data:
        lines = block.split('\n')
        for line in lines:
            if line.startswith('Question:'):
                question = line.split('Question:')[1].strip()
                questions.append(question)
            elif line.startswith('Answer:'):
                answer = line.split('Answer:')[1].strip()
                answers.append(answer)
                # Convert answers to numerical labels for simplicity
                labels.append(0 if answer == 'A' else 1)
            elif line.startswith('Related Images:'):
                image_path = line.split('Related Images:')[1].strip()
                images.append(image_path)

    # Preprocess text data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_masks = [], []
    for q, a in zip(questions, answers):
        encoded = tokenizer.encode_plus(q + " [SEP] " + a, add_special_tokens=True, max_length=512, pad_to_max_length=True, return_attention_mask=True, return_tensors='np')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = np.array(input_ids).squeeze()
    attention_masks = np.array(attention_masks).squeeze()

    # Preprocess image data
    def preprocess_image(image_path, target_size=(224, 224)):
        if image_path:
            img = Image.open(image_path).resize(target_size)
            img = np.array(img) / 255.0
        else:
            img = np.zeros(target_size + (3,))
        return img

    image_data = np.array([preprocess_image(img) for img in images])

    # Prepare labels
    label_data = to_categorical(labels, num_classes=2)

    return input_ids, attention_masks, image_data, label_data

# Load and preprocess the data
file_path = 'output/questions_answers_formatted.txt'  # Update this to your data file path
input_ids, attention_masks, image_data, label_data = load_and_preprocess_data(file_path)
# Load pre-trained BERT model and tokenizer
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Load pre-trained image model (e.g., ResNet)
image_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", trainable=False)

# Define model inputs
text_input = Input(shape=(512,), dtype='int32', name='text_input')
mask_input = Input(shape=(512,), dtype='int32', name='mask_input')
image_input = Input(shape=(224, 224, 3), name='image_input')

# Extract features
text_features = bert_model(text_input, attention_mask=mask_input)[1]
image_features = image_model(image_input)

# Combine features and add classification layers
combined_features = Concatenate()([text_features, image_features])
combined_features = Dense(1024, activation='relu')(combined_features)
output = Dense(2, activation='softmax')(combined_features)

# Build and compile the model
model = Model(inputs=[text_input, mask_input, image_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Split the data
train_inputs, test_inputs, train_masks, test_masks, train_images, test_images, train_labels, test_labels = train_test_split(input_ids, attention_masks, image_data, label_data, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices(({"text_input": train_inputs, "mask_input": train_masks, "image_input": train_images}, train_labels)).batch(16)
test_dataset = tf.data.Dataset.from_tensor_slices(({"text_input": test_inputs, "mask_input": test_masks, "image_input": test_images}, test_labels)).batch(16)

# Train the model
history = model.fit(train_dataset, epochs=5, validation_data=test_dataset)


# Save the model
model_save_path = 'output/model.h5'  # Specify the directory to save your model
model.save(model_save_path)
