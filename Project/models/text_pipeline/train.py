#creating csv file
import os
import pandas as pd

# 1. Update this path to exactly where your downloaded TESS folders are located in your Drive
# You can right-click the folder in that left pane and select "Copy path"
TESS_AUDIO_DIR = '/content/drive/MyDrive/IIITH 1M DATASET/speech dataset unzipped/TESS Toronto emotional speech set data'

# 2. Where you want to save the new CSV file
CSV_SAVE_PATH = '/content/drive/MyDrive/tess_text_data.csv'

data = []

print("Scanning audio files to generate text transcripts...")

# Walk through all folders and files in the TESS directory
for root, dirs, files in os.walk(TESS_AUDIO_DIR):
    for file in files:
        if file.endswith('.wav'):
            # TESS filenames are formatted like: OAF_dog_angry.wav
            # We split the filename by the underscore '_'
            parts = file.replace('.wav', '').split('_')

            # Ensure the filename actually has the 3 expected parts
            if len(parts) >= 3:
                speaker = parts[0]   # e.g., OAF or YAF
                word = parts[1]      # e.g., dog, back, base
                emotion = parts[2]   # e.g., angry, happy, sad

                # Reconstruct the transcript based on TESS documentation
                sentence = f"say the word {word}"

                data.append({
                    'filename': file,
                    'text': sentence,
                    'emotion': emotion
                })

# Convert the list to a pandas DataFrame
df = pd.DataFrame(data)

# Save it as a CSV
df.to_csv(CSV_SAVE_PATH, index=False)

print(f"\nSuccess! Created CSV with {len(df)} rows.")
print(f"Saved to: {CSV_SAVE_PATH}")

# Show a quick preview
df.head()

#text train.py
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# --- Configuration ---
DATA_PATH = '/content/drive/MyDrive/tess_text_data.csv' # Update this to your CSV path
MODEL_SAVE_PATH = '/content/drive/MyDrive/text_model.h5'
TOKENIZER_SAVE_PATH = '/content/drive/MyDrive/tokenizer.pickle'
LABEL_ENCODER_PATH = '/content/drive/MyDrive/label_encoder.pickle'
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 50  # Adjust based on your EDA of text lengths
EMBEDDING_DIM = 100       # Token x Features dimension

def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    # 1. PREPROCESSING BLOCK
    # Clean and tokenize text
    texts = df['text'].astype(str).tolist()
    labels = df['emotion'].tolist()

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    categorical_labels = to_categorical(encoded_labels)

    # Tokenization
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Padding to handle varying lengths
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Save Tokenizer and Label Encoder for testing
    with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(LABEL_ENCODER_PATH, 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return padded_sequences, categorical_labels, tokenizer.word_index, le.classes_

def build_model(vocab_size, num_classes):
    # Input Layer
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), name="text_input")

    # 2. FEATURE EXTRACTION BLOCK
    # Extracts emotional cues (tokens x features)
    embedding_layer = Embedding(input_dim=vocab_size,
                                output_dim=EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                name="feature_extraction_embedding")(input_layer)

    # 3. CONTEXTUAL MODELLING BLOCK
    # BiLSTM learns emotional meaning across tokens in context
    contextual_layer = Bidirectional(LSTM(64, return_sequences=False),
                                     name="contextual_modelling_bilstm")(embedding_layer)
    contextual_layer = Dropout(0.5)(contextual_layer)

    # 4. CLASSIFIER BLOCK
    # Predicts emotion label from learned representations
    classifier_dense = Dense(32, activation='relu', name="classifier_dense")(contextual_layer)
    output_layer = Dense(num_classes, activation='softmax', name="emotion_label_output")(classifier_dense)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    os.makedirs('../models', exist_ok=True)

    print("Loading and preprocessing data...")
    X, y, word_index, classes = load_and_preprocess_data()

    # Train/Test Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vocab_size = min(MAX_VOCAB_SIZE, len(word_index) + 1)

    print("Building model architecture...")
    model = build_model(vocab_size, len(classes))
    model.summary() # Great to include in your Report!

    print("Training model...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=32)

    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Training complete.")
