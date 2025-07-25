import torch
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType
import os
import gc
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# -----------------------------
# Data Loading
# -----------------------------
def load_data(file_path, is_train=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                label = 1 if item.get('label', '').upper() == "SARCASM" else 0 
                context = item.get('context', [])
                if not isinstance(context, list):
                    context = [str(context)]
                context = [str(c) for c in context]
                data.append({
                    'context': context,
                    'response': str(item.get('response', '')),
                    'label': label,
                    'id': item.get('id', '')
                })
            except:
                continue
    return pd.DataFrame(data)

# -----------------------------
# Dataset
# -----------------------------
class HierarchicalSarcasmDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_utterances=5, max_len=100):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_utterances = max_utterances
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        context = item['context']
        response = item['response']

        padded_context = context[:self.max_utterances]
        if len(padded_context) < self.max_utterances:
            padded_context += [''] * (self.max_utterances - len(padded_context))

        context_tokens = []
        for utt in padded_context:
            encoded = self.tokenizer(utt, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
            context_tokens.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })

        response_tokens = self.tokenizer(response, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')

        context_input_ids = torch.stack([t['input_ids'] for t in context_tokens])
        context_attention_mask = torch.stack([t['attention_mask'] for t in context_tokens])

        return {
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'response_input_ids': response_tokens['input_ids'].squeeze(0),
            'response_attention_mask': response_tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'id': item['id']
        }

# -----------------------------
# Model
# -----------------------------
class HierarchicalRoBERTaWithPEFT(nn.Module):
    def __init__(self, model_name='roberta-base', num_classes=2, lstm_hidden=128):
        super().__init__()
        peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "key", "value"])

        self.context_roberta = get_peft_model(RobertaModel.from_pretrained(model_name), peft_config)
        self.response_roberta = get_peft_model(RobertaModel.from_pretrained(model_name), peft_config)

        self.context_cnn = nn.Conv2d(1, 32, kernel_size=(2, 768))
        self.context_lstm = nn.LSTM(32, lstm_hidden, bidirectional=True, batch_first=True)
        self.context_attn = nn.MultiheadAttention(embed_dim=lstm_hidden*2, num_heads=4, batch_first=True)

        self.response_lstm = nn.LSTM(768, lstm_hidden, bidirectional=True, batch_first=True)
        self.response_attn = nn.MultiheadAttention(embed_dim=lstm_hidden*2, num_heads=4, batch_first=True)

        self.interaction_convs = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=(2, 2)),
            nn.Conv2d(1, 32, kernel_size=(2, 3)),
            nn.Conv2d(1, 32, kernel_size=(2, 5))
        ])

        self.classifier = nn.Sequential(
            nn.Linear(32 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, context_input_ids, context_attention_mask, response_input_ids, response_attention_mask):
        batch_size, num_utterances, seq_len = context_input_ids.shape

        context_embs = []
        for i in range(num_utterances):
            output = self.context_roberta(input_ids=context_input_ids[:, i, :], attention_mask=context_attention_mask[:, i, :]).last_hidden_state[:, 0, :]
            context_embs.append(output)

        context_stack = torch.stack(context_embs, dim=1).unsqueeze(1)
        cnn_out = F.relu(self.context_cnn(context_stack)).squeeze(-1)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.context_lstm(cnn_out)
        attn_out, _ = self.context_attn(lstm_out, lstm_out, lstm_out)
        context_repr = attn_out.mean(dim=1)

        response_output = self.response_roberta(input_ids=response_input_ids, attention_mask=response_attention_mask).last_hidden_state
        lstm_resp, _ = self.response_lstm(response_output)
        attn_resp, _ = self.response_attn(lstm_resp, lstm_resp, lstm_resp)
        response_repr = attn_resp.mean(dim=1)

        interaction_matrix = torch.cat([context_repr.unsqueeze(1), response_repr.unsqueeze(1)], dim=1).unsqueeze(1)
        conv_outputs = [F.max_pool2d(F.relu(conv(interaction_matrix)), kernel_size=(1, conv(interaction_matrix).shape[-1])).squeeze() for conv in self.interaction_convs]
        final_features = torch.cat(conv_outputs, dim=1)

        logits = self.classifier(final_features)
        return logits

# -----------------------------
# Training and Evaluation
# -----------------------------
if device.type == 'cuda' or device.type == 'mps':
    scaler = torch.cuda.amp.GradScaler()
else:
    class DummyScaler:
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass
        def unscale_(self, optimizer): pass
    scaler = DummyScaler()

def train(model, dataloader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        context_ids = batch['context_input_ids'].to(device)
        context_mask = batch['context_attention_mask'].to(device)
        response_ids = batch['response_input_ids'].to(device)
        response_mask = batch['response_attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
            outputs = model(context_ids, context_mask, response_ids, response_mask)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, preds, labels_all = 0, [], []

    with torch.no_grad():
        for batch in dataloader:
            context_ids = batch['context_input_ids'].to(device)
            context_mask = batch['context_attention_mask'].to(device)
            response_ids = batch['response_input_ids'].to(device)
            response_mask = batch['response_attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(context_ids, context_mask, response_ids, response_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    f1 = f1_score(labels_all, preds, zero_division=1)
    acc = accuracy_score(labels_all, preds)
    print(classification_report(labels_all, preds))
    return total_loss / len(dataloader), acc, f1

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    print("\n Hierarchical RoBERTa with PEFT initialized.")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    reddit_data = load_data('/sarcasm_detection_shared_task_reddit_training.jsonl')
    twitter_data = load_data('/sarcasm_detection_shared_task_twitter_training.jsonl')

    combined_data = pd.concat([reddit_data, twitter_data], ignore_index=True)

    train_val_data, test_data = train_test_split(
        combined_data, test_size=0.1, stratify=combined_data['label'], random_state=42
    )

    train_df, val_df = train_test_split(
        train_val_data, test_size=0.1, stratify=train_val_data['label'], random_state=42
    )

    #Debug Statements
    print("Train label dist:", train_df['label'].value_counts())
    print("Val label dist:", val_df['label'].value_counts())
    print("Test label dist:", test_data['label'].value_counts())

    train_dataset = HierarchicalSarcasmDataset(train_df, tokenizer)
    val_dataset = HierarchicalSarcasmDataset(val_df, tokenizer)
    test_dataset = HierarchicalSarcasmDataset(test_data, tokenizer)

    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = HierarchicalRoBERTaWithPEFT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

    print("\n Training ...")
    for epoch in range(3):
        train_loss = train(model, train_loader, optimizer, criterion, scheduler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

    print("\n Final Test Evaluation")
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
    torch.save(model.state_dict(), "hierarchical_roberta_sarcasm_model.pt")
    print(f"Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    print("Model saved!")
