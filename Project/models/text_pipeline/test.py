#text test.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
TEST_DATA_PATH = '/content/drive/MyDrive/tess_text_data.csv' # Or a dedicated test CSV if you split them beforehand
MODEL_PATH = '../models/text_model.h5'
TOKENIZER_PATH = '../models/tokenizer.pickle'
LABEL_ENCODER_PATH = '../models/label_encoder.pickle'
RESULTS_DIR = '../Results/'

MAX_SEQUENCE_LENGTH = 50

def load_test_data():
    df = pd.read_csv(TEST_DATA_PATH)
    # Ideally, you'd load just your holdout test set here.
    # For this script, we'll load the data and you can adjust to your specific test set.
    return df['text'].astype(str).tolist(), df['emotion'].tolist()

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading models and preprocessors...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open(LABEL_ENCODER_PATH, 'rb') as handle:
        le = pickle.load(handle)

    texts, true_labels = load_test_data()

    print("Preprocessing test data...")
    sequences = tokenizer.texts_to_sequences(texts)
    X_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    y_test_encoded = le.transform(true_labels)

    print("Making predictions...")
    predictions = model.predict(X_test)
    y_pred_encoded = np.argmax(predictions, axis=1)
    y_pred_labels = le.inverse_transform(y_pred_encoded)

    # 1. Classification Report (Accuracy Table requirement)
    print("\n--- Classification Report ---")
    report = classification_report(true_labels, y_pred_labels, target_names=le.classes_)
    print(report)

    # Save report to Results/
    with open(os.path.join(RESULTS_DIR, 'text_model_report.txt'), 'w') as f:
        f.write("Text-Only Model Classification Report\n")
        f.write("="*40 + "\n")
        f.write(report)

    # 2. Confusion Matrix (For Error Analysis: failure cases)
    cm = confusion_matrix(true_labels, y_pred_labels, labels=le.classes_)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Text Only Model')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'text_confusion_matrix.png'))
    print(f"\nSaved confusion matrix plot to {RESULTS_DIR}")
