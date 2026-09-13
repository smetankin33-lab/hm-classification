import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

print(f" PyTorch version: {torch.__version__}")

# Загрузка данных
df = pd.read_csv('../data/articles_cleaned.csv')
df['text'] = df['prod_name'].fillna('') + ' ' + df['detail_desc'].fillna('')


# Очистка текста
def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower())


X = df['text'].apply(clean_text)
y = df['garment_group_name']

# ТОТ ЖЕ сплит
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

classes = sorted(y.unique())
class_to_idx = {c: i for i, c in enumerate(classes)}

y_train_enc = np.array([class_to_idx[c] for c in y_train])
y_test_enc = np.array([class_to_idx[c] for c in y_test])

# ==========================================
# 1. Токенизация
# ==========================================
MAX_WORDS = 10000
MAX_LEN = 50  # ИСПРАВЛЕНИЕ: уменьшаем длину, описания товаров короткие


def build_vocab(texts, max_words):
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_words - 2]
    vocab = {word: i + 2 for i, (word, _) in enumerate(sorted_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab


vocab = build_vocab(X_train, MAX_WORDS)
print(f" Словарь построен. Размер: {len(vocab)}")


def text_to_sequence(text, vocab, max_len):
    seq = [vocab.get(w, 1) for w in text.split()]
    if len(seq) > max_len:
        seq = seq[-max_len:]  # ИСПРАВЛЕНИЕ: берём ПОСЛЕДНИЕ max_len слов
    else:
        # ИСПРАВЛЕНИЕ: PRE-PADDING — дополняем нулями В НАЧАЛЕ
        seq = [0] * (max_len - len(seq)) + seq
    return seq


# Проверим длину текстов
lengths = X_train.apply(lambda x: len(x.split()))
print(f" Статистика длины текстов (в словах):")
print(f"   Mean: {lengths.mean():.1f}, Median: {lengths.median():.1f}, Max: {lengths.max()}")
print(f"   Процент текстов <= 50 слов: {(lengths <= 50).mean() * 100:.1f}%")


# ==========================================
# 2. Dataset и DataLoader
# ==========================================
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, vocab=None, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        seq = text_to_sequence(text, self.vocab, self.max_len)
        item = {'input_ids': torch.tensor(seq, dtype=torch.long)}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = TextDataset(X_train, y_train_enc, vocab, MAX_LEN)
test_dataset = TextDataset(X_test, y_test_enc, vocab, MAX_LEN)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================
# 3. Модель (Embedding -> LSTM -> Dense)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx=0):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 из-за bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, (hidden, _) = self.lstm(embedded)
        # Берём hidden из обоих направлений
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1)
        hidden_cat = self.dropout(hidden_cat)
        out = self.fc(hidden_cat)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LSTMClassifier(len(vocab), embed_dim=128, hidden_dim=128, output_dim=len(classes)).to(device)

# Веса классов для борьбы с дисбалансом
class_counts = np.bincount(y_train_enc)
total = len(y_train_enc)
weights = total / (len(classes) * class_counts)
class_weights = torch.FloatTensor(weights).to(device)
print(f"⚖️ Веса классов: {dict(zip(classes, weights.round(2)))}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ==========================================
# 4. Обучение (6 эпох по ТЗ)
# ==========================================
EPOCHS = 6
train_losses = []

print(f"\n Начинаем обучение LSTM ({EPOCHS} эпох)...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total_samples = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        predictions = model(input_ids)
        loss = criterion(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        epoch_loss += loss.item() * input_ids.size(0)
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    scheduler.step()
    avg_train_loss = epoch_loss / total_samples
    train_acc = correct / total_samples
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

# ==========================================
# 5. График функции потерь
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, marker='o', label='Train Loss')
plt.title('Функция потерь (Loss) по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.xticks(range(1, EPOCHS + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 6. Оценка на тестовой выборке
# ==========================================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels']

        predictions = model(input_ids)
        _, predicted = torch.max(predictions, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

y_pred_enc = np.array(all_preds)
y_test_enc = np.array(all_labels)
y_pred_lstm = np.array([classes[i] for i in y_pred_enc])

lstm_acc = accuracy_score(y_test, y_pred_lstm)
lstm_f1 = f1_score(y_test, y_pred_lstm, average='macro')

print(f"\n--- LSTM Results ---")
print(f"Accuracy: {lstm_acc:.4f}")
print(f"Macro-F1: {lstm_f1:.4f}")

# ==========================================
# 7. Матрица ошибок LSTM
# ==========================================
cm_lstm = confusion_matrix(y_test, y_pred_lstm, labels=classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок (LSTM)')
plt.tight_layout()
plt.show()

# ==========================================
# 8. Анализ ошибок LSTM
# ==========================================
errors_mask = y_test.values != y_pred_lstm
errors_df = pd.DataFrame({
    'text': X_test[errors_mask],
    'true_class': y_test[errors_mask],
    'pred_class': y_pred_lstm[errors_mask]
})

print(f"\n Всего ошибок LSTM на тесте: {len(errors_df)}")
print("\n--- Примеры 3 ошибочных предсказаний LSTM ---")
errors_to_show = errors_df.head(3).copy()
errors_to_show['text'] = errors_to_show['text'].apply(lambda x: x[:100] + '...')
print(errors_to_show.to_string())

# ==========================================
# 9. Сохранение
# ==========================================
os.makedirs('../models', exist_ok=True)
torch.save(model.state_dict(), '../models/lstm_model.pth')

with open('../models/vocab.pickle', 'wb') as handle:
    pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../models/classes.json', 'w', encoding='utf-8') as f:
    json.dump(classes, f, ensure_ascii=False)

# Сохраняем параметры модели для загрузки в API
model_config = {
    'vocab_size': len(vocab),
    'embed_dim': 128,
    'hidden_dim': 128,
    'output_dim': len(classes),
    'max_len': MAX_LEN
}
with open('../models/model_config.json', 'w') as f:
    json.dump(model_config, f)

print("\n Модель LSTM, словарь, классы и конфиг сохранены в папке models/")