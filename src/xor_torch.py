import sys
import torch
import torch.nn.functional as F
from torch import nn

### Check if we want to load a model or save a new one.
load=False
if (len(sys.argv) >= 2 and sys.argv[1] == 'l'):
    load=True

# Get cpu, gpu or mps device for training.
device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
print(f"Using {device} device")

# Training data
# 0 XOR 0 = 0
# 1 XOR 0 = 1
# 0 XOR 1 = 1
# 1 XOR 1 = 0
training_data_input = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
training_data_target = torch.Tensor([[1, 0], [0, 1], [0, 1], [1, 0]]).to(device)

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 input nodes, 16 hidden nodes and 2 output nodes
        self.input = nn.Linear(2, 16)
        self.hidden = nn.Linear(16, 2)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)

    def forward(self, x):
        out = F.relu(self.input(x))
        return F.sigmoid(self.hidden(out))

    def load(self):
        print("Loading the model ...")
        self.load_state_dict(torch.load("xor.pth"))

    def save(self):
        torch.save(self.state_dict(), "xor.pth")
        print("Model was saved.")

model = NeuralNetwork().to(device)

# Train the neural network, or load a model
if (load):
    model.load()
else:
   model.init_weights()
   loss_fn = nn.BCELoss(reduction='mean')
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

   epochs = 10000
   for e in range(epochs):
       # Compute prediction error
       pred = model(training_data_input)
       loss = loss_fn(pred, training_data_target)

       # Backpropagation
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()

       if e % 500 == 0:
           print('Epoch: ', e, float(loss))

   model.save()

# Query the network
print(model(torch.Tensor([[0, 0]])).to(device))
print(model(torch.Tensor([[0, 1]])).to(device))
print(model(torch.Tensor([[1, 0]])).to(device))
print(model(torch.Tensor([[1, 1]])).to(device))
