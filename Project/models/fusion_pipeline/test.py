import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. DATASET & PREPROCESSING (Identical to train.py)
# ==========================================
class TESSSpeechTextDataset(Dataset):
    def __init__(self, data_dir, max_time_steps=100, max_tokens=20):
        self.data_dir = data_dir
        self.max_time_steps = max_time_steps
        self.max_tokens = max_tokens

        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'neutral': 4, 'ps': 5, 'sad': 6
        }

        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.data = self._load_data()

    def _load_data(self):
        data = []
        if not os.path.exists(self.data_dir):
            return data

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".wav"):
                    filepath = os.path.join(root, file)
                    parts = file.replace('.wav', '').split('_')
                    if len(parts) >= 3:
                        word = parts[1].lower()
                        emotion = parts[2].lower()

                        if emotion in self.emotion_map:
                            data.append({
                                'path': filepath,
                                'text': word,
                                'label': self.emotion_map[emotion]
                            })
                            if word not in self.vocab:
                                self.vocab[word] = len(self.vocab)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        y, sr = librosa.load(item['path'], sr=16000)
        y_trimmed, _ = librosa.effects.trim(y)
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=40).T

        if mfccs.shape[0] < self.max_time_steps:
            pad_width = self.max_time_steps - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:self.max_time_steps, :]

        word = item['text']
        token_id = self.vocab.get(word, self.vocab["<UNK>"])

        text_tokens = [token_id]
        if len(text_tokens) < self.max_tokens:
            text_tokens.extend([self.vocab["<PAD>"]] * (self.max_tokens - len(text_tokens)))

        return (
            torch.tensor(mfccs, dtype=torch.float32),
            torch.tensor(text_tokens, dtype=torch.long),
            torch.tensor(item['label'], dtype=torch.long)
        )

# ==========================================
# 2. MODEL ARCHITECTURES (Identical to train.py)
# ==========================================
class SpeechEmotionModel(nn.Module):
    def __init__(self, input_features, hidden_size, num_classes):
        super(SpeechEmotionModel, self).__init__()
        self.lstm = nn.LSTM(input_features, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hidden = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1)
        output = self.classifier(hidden)
        return output, hidden

class TextEmotionModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super(TextEmotionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        output = self.classifier(hidden)
        return output, hidden

class MultimodalEmotionModel(nn.Module):
    def __init__(self, speech_model, text_model, num_classes):
        super(MultimodalEmotionModel, self).__init__()
        self.speech_branch = speech_model
        self.text_branch = text_model
        self.speech_branch.classifier = nn.Identity()
        self.text_branch.classifier = nn.Identity()

        fusion_dim = 512
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, speech_x, text_x):
        _, speech_rep = self.speech_branch(speech_x)
        _, text_rep = self.text_branch(text_x)
        fused_representation = torch.cat((speech_rep, text_rep), dim=1)
        output = self.classifier(fused_representation)
        return output, fused_representation

# ==========================================
# 3. EVALUATION PIPELINE
# ==========================================
def main():
    data_path = '/content/drive/MyDrive/IIITH 1M DATASET/speech dataset unzipped'
    batch_size = 32
    num_emotions = 7

    print("Initializing Dataset...")
    dataset = TESSSpeechTextDataset(data_dir=data_path)

    if len(dataset) == 0:
        print("Exiting: No data found. Please check your data directory.")
        return

    # Fix the random seed to ensure we get the exact same test split as training
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing Models & Loading Weights...")
    speech_base = SpeechEmotionModel(input_features=40, hidden_size=128, num_classes=num_emotions)
    text_base = TextEmotionModel(vocab_size=len(dataset.vocab), embed_size=100, hidden_size=128, num_classes=num_emotions)
    fusion_model = MultimodalEmotionModel(speech_base, text_base, num_classes=num_emotions)

    # Load the saved weights from train.py
    try:
        fusion_model.load_state_dict(torch.load('fusion_model.pth'))
    except FileNotFoundError:
        print("Error: 'fusion_model.pth' not found. Please run the training script first.")
        return

    print("\nEvaluating Multimodal Fusion Model...")
    fusion_model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for speech_data, text_data, labels in test_loader:
            outputs, _ = fusion_model(speech_data, text_data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nFusion Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    target_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
    print(classification_report(all_labels, all_preds, target_names=target_names))

if __name__ == "__main__":
    main()
