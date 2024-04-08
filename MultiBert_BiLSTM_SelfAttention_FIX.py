import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# 加载数据
# file_path = '../../../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled_Example.csv'
file_path = '../../../datasets_FIX2/FIX2_deduplicated_mangoNews_Nums3000p_CategoryMerge_new_undersampled.csv'

data = pd.read_csv(file_path,low_memory=False,lineterminator="\n")

texts = data['body'].tolist()
labels = data['category1'].tolist()

# 对标签进行编码
unique_labels = list(set(labels))
label_to_id = {label: i for i, label in enumerate(unique_labels)}
labels = [label_to_id[label] for label in labels]
num_classes = len(unique_labels)

# # 加载BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('../../bert-base-multilingual-cased')

# 加载BERT tokenizer和模型
model_name = '../../../bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# 定义数据集
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
# 定义Self-Attention层
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = torch.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs

# 定义模型
class NewsClassifier(nn.Module):
    # hidden_size = 128
    def __init__(self, num_classes, hidden_size, num_layers=2, bidirectional=True):
        super(NewsClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained('../../bert-base-multilingual-cased')
        self.bert = bert_model # FIX

        # self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, 
        #                     bidirectional=bidirectional, batch_first=True)
        self.lstm = nn.LSTM(input_size=bert_model.config.hidden_size, hidden_size=hidden_size, num_layers=num_layers, 
                            bidirectional=bidirectional, batch_first=True) # FIX
        
        self.attention = SelfAttention(hidden_size * (2 if bidirectional else 1))
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        lstm_outputs, _ = self.lstm(last_hidden_state)
        attention_outputs = self.attention(lstm_outputs)
        logits = self.fc(attention_outputs)
        return logits

# 定义训练函数
def train_model(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        # progress_bar.set_postfix({'Loss': avg_loss:.4f})
        # progress_bar.set_postfix({'Loss': avg_loss:.4f})
        # progress_bar.set_postfix({'Loss': "{:.4f}".format(avg_loss) })
        progress_bar.set_postfix({'loss': loss.item()})
        # progress_bar.set_postfix({'Loss': "{:.4f}".format(loss.item) }) # FIX

    return avg_loss

# 定义评估函数
def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            batch_predictions = torch.argmax(logits, dim=1)

            predictions.extend(batch_predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return classification_report(true_labels, predictions, digits=4)

# 设置超参数
# max_length = 256
max_length = 512
# origin:max_length = 512
batch_size = 16
epochs = 20
learning_rate = 2e-5
# hidden_size = 768
hidden_size = 128
# origin：hidden_size = 128

num_layers = 2
bidirectional = True

# 使用KFold进行交叉验证
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed) # FIX


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_models = []
best_reports = []

for fold, (train_index, val_index) in enumerate(kfold.split(texts, labels)):
    print(f'Fold {fold + 1}')
    print('-' * 30)

    train_texts, val_texts = [texts[i] for i in train_index], [texts[i] for i in val_index]
    train_labels, val_labels = [labels[i] for i in train_index], [labels[i] for i in val_index]

    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = NewsClassifier(num_classes, hidden_size, num_layers, bidirectional)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        train_loss = train_model(model, train_dataloader, optimizer, scheduler, device, epoch)
        val_report = evaluate_model(model, val_dataloader, device)

        val_loss = 1 - float(val_report.split('\n')[-3].split()[-2])  # 提取验证集损失

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}')
        print('Validation Report:')
        print(val_report)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # best_model = model.state_dict()
            # best_report = val_report
            torch.save(model.state_dict(), f'best_MultiBert_BiLSTM_SelfAttention_modelFIX_fold_{fold + 1}.pth')


# 在每个fold结束后,评估最佳模型在验证集上的性能
    best_model = NewsClassifier(num_classes, hidden_size, num_layers, bidirectional)

    best_model.load_state_dict(torch.load(f'best_MultiBert_BiLSTM_SelfAttention_modelFIX_fold_{fold + 1}.pth'))
    best_model.to(device)
    val_report = evaluate_model(best_model, val_dataloader, device)
    # all_reports.append(val_report)

    print(f'Fold {fold + 1} Best Validation Report:')
    print(val_report)
    print()

