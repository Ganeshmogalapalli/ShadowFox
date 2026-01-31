# ==========================================
# INSTALL REQUIRED PACKAGES
# pip install torch transformers datasets scikit-learn
# ==========================================

import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from  transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ------------------------------------------
# SYSTEM CONFIG (CPU SAFE)
# ------------------------------------------
os.environ["OMP_NUM_THREADS"] = "2"
torch.set_num_threads(2)

DEVICE = torch.device("cpu")

# ------------------------------------------
# DATA WRAPPER
# ------------------------------------------
class IMDbData:
    def __init__(self, train_size=500, test_size=100):
        raw = load_dataset("imdb")
        self.train = raw["train"].shuffle(seed=7).select(range(train_size))
        self.test = raw["test"].shuffle(seed=7).select(range(test_size))

# ------------------------------------------
# CUSTOM DATASET
# ------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.data = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.data.items()}
        item["label"] = torch.tensor(self.labels[idx])
        return item

# ------------------------------------------
# MODEL DEFINITION (NO CLASSIFIER HEAD USED)
# ------------------------------------------
class SentimentNet(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)

# ------------------------------------------
# TRAINING ENGINE
# ------------------------------------------
class TrainerEngine:
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim = torch.optim.AdamW(model.parameters(), lr=3e-5)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for batch in loader:
            self.optim.zero_grad()

            logits = self.model(
                batch["input_ids"],
                batch["attention_mask"]
            )

            loss = self.loss_fn(logits, batch["label"])
            loss.backward()
            self.optim.step()

            total_loss += loss.item()

        return total_loss

# ------------------------------------------
# EVALUATOR
# ------------------------------------------
class Evaluator:
    @staticmethod
    def evaluate(model, loader):
        model.eval()
        preds, gold = [], []

        with torch.no_grad():
            for batch in loader:
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"]
                )
                preds.extend(torch.argmax(logits, 1).tolist())
                gold.extend(batch["label"].tolist())

        return accuracy_score(gold, preds)

# ------------------------------------------
# PREDICTOR
# ------------------------------------------
class Predictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, text):
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            logits = self.model(
                tokens["input_ids"],
                tokens["attention_mask"]
            )

        probs = torch.softmax(logits, dim=1)
        idx = torch.argmax(probs).item()

        return ("POSITIVE" if idx else "NEGATIVE", probs[0][idx].item())

# ------------------------------------------
# MAIN EXECUTION
# ------------------------------------------
def main():
    print("Loading data...")
    data = IMDbData()

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    train_set = TextDataset(data.train["text"], data.train["label"], tokenizer)
    test_set = TextDataset(data.test["text"], data.test["label"], tokenizer)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8)

    model = SentimentNet("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)

    trainer = TrainerEngine(model)

    print("\nTraining...")
    for epoch in range(2):
        loss = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f}")

    print("\nEvaluating...")
    acc = Evaluator.evaluate(model, test_loader)
    print(f"Test Accuracy: {acc:.4f}")

    predictor = Predictor(model, tokenizer)

    print("\nCustom Predictions:")
    tests = [
        "This movie was brilliant and touching",
        "Completely waste of time",
        "It was okay, nothing extraordinary"
    ]

    for t in tests:
        label, conf = predictor.predict(t)
        print(f"{t}\n→ {label} ({conf:.2f})\n")

if __name__ == "__main__":
    main()
