import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, classification_report
import os

data_dir = "FakeNewsNet"


def load_data():
    fake_buzz = pd.read_csv(os.path.join(data_dir, "BuzzFeed_fake_news_content.csv"))
    real_buzz = pd.read_csv(os.path.join(data_dir, "BuzzFeed_real_news_content.csv"))
    fake_politifact = pd.read_csv(os.path.join(data_dir, "PolitiFact_fake_news_content.csv"))
    real_politifact = pd.read_csv(os.path.join(data_dir, "PolitiFact_real_news_content.csv"))

    fake_buzz["label"] = 1
    real_buzz["label"] = 0
    fake_politifact["label"] = 1
    real_politifact["label"] = 0

    df = pd.concat([fake_buzz, real_buzz, fake_politifact, real_politifact], ignore_index=True)

    title_column = "title" if "title" in df.columns else df.columns[0]

    return df[[title_column, "label"]]


df = load_data()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        return encoding.input_ids.squeeze(0), encoding.attention_mask.squeeze(0), torch.tensor(label)


texts = df.iloc[:, 0].tolist()
labels = df["label"].tolist()

dataset = NewsDataset(texts, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)


def train_model(model, train_loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input_ids, attention_mask, labels in train_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


def evaluate_model(model, test_loader):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())
    print("Accuracy:", accuracy_score(true_labels, predictions))
    print("Classification Report:\n", classification_report(true_labels, predictions))


train_model(model, train_loader, optimizer, criterion, epochs=3)
evaluate_model(model, test_loader)
