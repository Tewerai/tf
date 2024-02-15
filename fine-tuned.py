import re
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Concatenate, LSTM, Embedding
from tensorflow.keras.models import Model
import tensorflow_hub as hub
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().split('\n\n')  # Split data into blocks

    questions, explanations, images, labels = [], [], [], []
    for block in data:
        lines = block.split('\n')
        for line in lines:
            if line.startswith('Question:'):
                question = line.split('Question:')[1].strip()
                questions.append(question)
            elif line.startswith('Answer:'):
                answer = line.split('Answer:')[1].strip()
                labels.append(0 if answer == 'A' else 1)
            elif line.startswith('Related Images:'):
                image_path = line.split('Related Images:')[1].strip()
                images.append(image_path)
            elif line.startswith('Explanation:'):
                explanation = line.split('Explanation:')[1].strip()
                explanations.append(explanation)

    # Preprocess text data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_masks = [], []
    for q in questions:
        encoded = tokenizer.encode_plus(q, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, return_tensors='np')
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = np.array(input_ids).squeeze()
    attention_masks = np.array(attention_masks).squeeze()

    # Preprocess image data
    image_data = np.array([preprocess_image(img) for img in images])

    # Prepare labels
    label_data = to_categorical(labels, num_classes=2)

    # Encode explanations
    explanation_data = encode_explanations(explanations, tokenizer)

    return input_ids, attention_masks, image_data, label_data, explanation_data

def preprocess_image(image_path, target_size=(224, 224)):
    if image_path:
        img = Image.open(image_path).resize(target_size)
        img = np.array(img) / 255.0
    else:
        img = np.zeros(target_size + (3,))
    return img

def encode_explanations(explanations, tokenizer):
    max_length = 128  # Adjust based on your dataset
    encoded_explanations = [tokenizer.encode(exp, max_length=max_length, padding='max_length', truncation=True) for exp in explanations]
    return np.array(encoded_explanations)

# Load and preprocess the data
file_path = 'output/questions_answers_explanations.txt'  # Update this to your data file path
input_ids, attention_masks, image_data, label_data, explanation_data = load_and_preprocess_data(file_path)

# Define the model
def build_model():
    # Load pre-trained BERT model and tokenizer
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')

    # Load pre-trained image model (e.g., ResNet)
    image_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4", trainable=False)

    # Model inputs
    text_input = Input(shape=(512,), dtype='int32', name='text_input')
    mask_input = Input(shape=(512,), dtype='int32', name='mask_input')
    image_input = Input(shape=(224, 224, 3), name='image_input')

    # Extract features
    text_features = bert_model(text_input, attention_mask=mask_input)[1]
    image_features = image_model(image_input)

    # Combine features
    combined_features = Concatenate()([text_features, image_features])

    # Answer prediction
    answer_output = Dense(2, activation='softmax', name='answer_output')(combined_features)

    # Explanation generation
    explanation_input = Dense(512, activation='relu')(combined_features)  # Adapt size as needed
    explanation
