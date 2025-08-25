import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 1. 数据加载与预处理（保持不变）
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40


# 2. Dataset 类（保持不变）
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        bow_vectors = []
        for text in self.texts:
            # Tokenize and truncate/pad
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            # Create BoW vector
            bow_vector = torch.zeros(self.vocab_size)
            for idx in tokenized:
                if idx != 0:
                    bow_vector[idx] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# 构建数据集
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)


# 3. 不同结构的模型定义
def create_model(config, input_dim, output_dim):
    layers = []
    in_dim = input_dim
    for hidden_dim in config:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    # 输出层
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


# 4. 训练与记录函数
def train_model(model, dataloader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    loss_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return loss_history



# 5. 执行
input_dim = vocab_size
output_dim = len(label_to_index)
num_epochs = 10

# 定义不同模型结构（隐藏层配置）
experiments = {
    "Exp1: 128": [128],
    "Exp2: 64": [64],
    "Exp3: 256": [256],
    "Exp4: 128→64": [128, 64],
    "Exp5: 256→128→64": [256, 128, 64]
}

results = {}

for name, hidden_config in experiments.items():
    print(f"开始训练模型: {name}")
    model = create_model(hidden_config, input_dim, output_dim)
    loss_hist = train_model(model, dataloader, num_epochs)
    results[name] = loss_hist