#speech train.py
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 1. Preprocessing & Feature Extraction
def extract_audio_features(file_path, max_len=150):
    """
    Loads audio, trims silence, and extracts MFCC features over time.
    Returns shape: (time_steps, features) to feed into LSTM.
    """
    try:
        # Load audio and choose sampling rate (16kHz is standard for speech)
        audio, sr = librosa.load(file_path, sr=16000)

        # Trim leading and trailing silence
        audio, _ = librosa.effects.trim(audio)

        # Extract MFCCs (Extracting 40 features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        # Transpose so time is the first dimension: (time_steps, features)
        mfccs = mfccs.T

        # Handle varying lengths by padding or truncating to max_len
        if mfccs.shape[0] > max_len:
            mfccs = mfccs[:max_len, :]
        else:
            pad_width = max_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, pad_width=((0, pad_width), (0, 0)), mode='constant')

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def prepare_dataset(dataset_path):
    X = []
    y = []

    print("Extracting features from audio files...")
    # TESS dataset structure: Folders are named like 'OAF_angry', 'YAF_sad'
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Extract the emotion label from the folder name (e.g., 'angry' from 'OAF_angry')
        emotion_label = folder_name.split('_')[-1].lower()

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                features = extract_audio_features(file_path)

                if features is not None:
                    X.append(features)
                    y.append(emotion_label)

    return np.array(X), np.array(y)

# 2. PyTorch Dataset Definition
class TESSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


#3. Temporal Modelling & Classifier (LSTM)

class SpeechEmotionLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=7):
        super(SpeechEmotionLSTM, self).__init__()

        # Temporal Modelling: Learn emotional patterns over time
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Classifier: Predict emotion from learned representations
        # hidden_size * 2 because it's a bidirectional LSTM
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, time_steps, features)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Extract the final hidden state from both directions
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        final_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)

        # Pass through classifier
        out = self.classifier(final_hidden)
        return out


# 4. Main Execution & Training Loop

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    # Update this path if your extracted folder is named differently
    DATASET_PATH = "/content/drive/MyDrive/IIITH 1M DATASET/speech dataset unzipped/TESS Toronto emotional speech set data"

    if os.path.exists(DATASET_PATH):
        X, y_text = prepare_dataset(DATASET_PATH)

        # Encode string labels to integers
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_text)
        num_classes = len(label_encoder.classes_)
        print(f"Classes found: {label_encoder.classes_}")

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create DataLoaders
        train_dataset = TESSDataset(X_train, y_train)
        test_dataset = TESSDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 2. Initialize Model, Loss, and Optimizer
        model = SpeechEmotionLSTM(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 3. Training Loop
        epochs = 30
        print("\nStarting Training...")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total

            # Validation
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_acc = 100 * test_correct / test_total

            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")

        print("\nTraining Complete!")
    else:
        print(f"Dataset path '{DATASET_PATH}' not found. Please upload and extract the TESS dataset to Colab.")
        # --- GUARANTEED DRIVE SAVE ---
        # This forces the file directly into your main Google Drive folder
        MODEL_SAVE_PATH = "/content/drive/MyDrive/speech_model.pth"

        # Save the weights
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"\nModel safely saved to your Drive at: {MODEL_SAVE_PATH}")
        print("\nSaving model to Google Drive...")
        import os
        from google.colab import drive

        # 1. Force Colab to connect to your Drive
        drive.mount('/content/drive', force_remount=True)

        # 2. Define the permanent path
        MODEL_SAVE_PATH = "/content/drive/MyDrive/speech_model.pth"

        # 3. Save the PyTorch weights
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # 4. Verify the file actually exists
        if os.path.exists(MODEL_SAVE_PATH):
            print(f"✅ SUCCESS: Speech model safely saved to {MODEL_SAVE_PATH}")
            print("You can now run your Multimodal Fusion script!")
        else:
            print("❌ ERROR: The model failed to save to Drive.")
          
