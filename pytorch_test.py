import torch
from torch import nn

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit] 

    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices) # suffle all the index of the data
    
    x, y = x[all_indices], y[all_indices] 
    x = x.reshape(len(x), 1, 28, 28) # reshape to 4-D tensor
    x = x.astype("float32") / 255 # standardize the output in [0,1]
    
    # one-hot encoder for the output
    y = to_categorical(y)
    y = y.reshape(len(y), 2)

    return x, y

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# Transfer x_train, y_train to torch.Tensor
x_train = torch.tensor(x_train, dtype=torch.float32)  
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(5 * 26 * 26, 100),
            nn.Sigmoid(),
            nn.Linear(100, 2),
            nn.Sigmoid()
            )
        
    def forward(self,x):
        return self.model(x)

# Create random seed
torch.manual_seed(42)

# Create model instance
model_0 = Model()

# Setup a loss function
loss_fn = nn.BCELoss()

# Setup an optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1) # use stochastic gradient descent

# Epochs
epochs = 20

# 0. Loop through the data
for epoch in range(epochs):
    model_0.train() # cho mô hình vào trạng thái train: set các param yêu cầu gradients 
    
    # 1. Forward pass
    y_preds = model_0(x_train)

    # 2. Calculate loss
    loss = loss_fn(y_preds, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

     # 6. accuracy
    with torch.no_grad():
        y_pred_labels = y_preds.argmax(dim=1)
        y_true_labels = y_train.argmax(dim=1)
        acc = (y_pred_labels == y_true_labels).float().mean().item()
    
    # 7. In thông tin mỗi epoch
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - Accuracy: {acc*100:.5f}%")
    
correct_pred = 0
for x, y in zip(x_test, y_test):
    x = x.unsqueeze(0)   
    output=model_0(x)
    if output.argmax() == y.argmax():
        correct_pred += 1
print(f"\nAccuracy on the test data: {correct_pred/len(y_test)*100:.5f}%")