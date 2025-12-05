import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import random

# ========================= CONFIG =========================
MODEL_NAME = "roberta-base"          
DATA_FILE = "classification_results_filtered.json"     
BATCH_SIZE = 32
MAX_LENGTH = 256
EPOCHS = 4
LR = 2e-5
SAVE_PATH = "save_model\discriminatorRo\discriminator_trained.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================================

# 1. Load dataset (supports both .json list and .jsonl)
def load_dataset(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        texts.append(item["prompt"])
        labels.append(int(item["predicted_label"]))
    return texts, labels

texts, labels = load_dataset(DATA_FILE)
print(f"Loaded {len(texts)} samples → Real: {sum(labels)}, Fake: {len(labels)-sum(labels)}")

# 2. Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }

# 3. Discriminator (same strong one)
class TextDiscriminator(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)   # ← raw logits

    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]       # [CLS]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)   
        return logits                                  

# 4. Setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = TextDataset(texts, labels, tokenizer, MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TextDiscriminator().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()
scaler = GradScaler()

# 5. Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        with torch.no_grad():
            predicted = (logits > 0.0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc*100:5.2f}%")

# 6. Save
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH.replace(".pth", "_tokenizer"))


