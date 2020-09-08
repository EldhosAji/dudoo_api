import random
import json
import torch
from .model import NeuralNet
from .utils import TOCKENIZE
from .utils import WORD_CONTAINER
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)),'ai\\intents.json'),'r') as f:
    intents=json.load(f)

FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),"ai\\data.ai")
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
WORD_LIST = data["WORD_LIST"]
TAGS = data["TAGS"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)

model.load_state_dict(model_state)
model.eval()

ai_name = "DuDoo"

def ai_chat(query):
    sentance = query
    sentance = TOCKENIZE(sentance)

    X = WORD_CONTAINER(sentance,WORD_LIST)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _,predicted = torch.max(output,dim=1)
    tag = TAGS[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return (random.choice(intent["responses"]))
    else :
        return ("Sorry,I'm programmed to answer only some of the questions")