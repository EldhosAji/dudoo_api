import json
import numpy as np
from utils import TOCKENIZE,STEM
from utils import WORD_CONTAINER
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet
with open('intents.json','r') as f:
    intents = json.load(f)

WORD_LIST = []

TAGS = []

xy =[]

for intent in intents['intents']:
    tag = intent['tag']
    TAGS.append(tag)
    for pattern in intent['patterns']:
        w = TOCKENIZE(pattern)
        WORD_LIST.extend(w)
        xy.append((w,tag))

ignore_words = ['?','!','.',',']

WORD_LIST = [STEM(w) for w in WORD_LIST if w not in ignore_words]

WORD_LIST = sorted(set(WORD_LIST))
TAGS = sorted(set(TAGS))

X_TRAIN = []
Y_TRAIN = []

for (pattern_sentence,tag) in xy:
    wordlist = WORD_CONTAINER(pattern_sentence,WORD_LIST)
    X_TRAIN.append(wordlist)

    label =TAGS.index(tag)
    Y_TRAIN.append(label)

X_TRAIN = np.array(X_TRAIN)
Y_TRAIN = np.array(Y_TRAIN)




batch_size = 8
output_size = len(TAGS)
input_size = len(X_TRAIN[0])
hidden_size = 8
lr = 0.001 #Learning_rate
num_epochs = 5000

class ChatDataset(Dataset):
    def __init__(self):
        self.n_sample=len(X_TRAIN)
        self.X_DATA = X_TRAIN
        self.Y_DATA= Y_TRAIN

    def __getitem__(self,index):
        return self.X_DATA[index],self.Y_DATA[index]

    def __len__(self):
        return self.n_sample


dataset = ChatDataset()

train_loader =DataLoader(dataset,batch_size,shuffle=True,num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size,hidden_size,output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1)%1000 ==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "WORD_LIST":WORD_LIST,
    "TAGS":TAGS
}

FILE = "data.ai"
torch.save(data,FILE)

print(f'Trainning completed -> {FILE}')