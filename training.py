import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_Functions import bag_of_words, tokenize, stem
from Layers_model import NeuralNet


with open('conversations.json', 'r') as f:
    convers = json.load(f)

all_words = []
tags = []
xy = []
#looping all our database conversations:
for intent in convers['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # we gonna  tokenize each word in the sentence
        w = tokenize(pattern)
        # adding tokenized worlds  to our all  words list
        all_words.extend(w)
        # add to xy list contains tokenzed words with their tag [word,tag]
        xy.append((w, tag))

#  using stemmer for each words
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# sorting and remove doubles in our list
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# traing input and output
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X train
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y train
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 3000
batch_size =69
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 69
output_size = len(tags)

#output and input size of model .
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # get index of data[]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

  #get lenght of data
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
      
        outputs = model(words)
        
        loss = criterion(outputs, labels)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

file = "model_trained.pth"
torch.save(data, file)

print(f' training finished  {file}')
