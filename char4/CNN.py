import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from char4.MFCLT import test_loader, train_loader

# Assuming input_size, hidden_size, and output_size are defined
input_size = 7
hidden_size = 64
output_size = 48


class SimpleCNN(nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool1d(kernel_size=2)
		self.flatten = nn.Flatten()
		self.fc = nn.Linear(32 * (output_size // 2), output_size)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.flatten(x)
		x = self.fc(x)
		return x


# Create instances of the models
cnn_model = SimpleCNN()

# Define your optimizer and criterion
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Assuming train_loader and test_loader are defined
num_epochs = 200
for epoch in range(num_epochs):
	cnn_model.train()
	for data, targets in train_loader:
		cnn_optimizer.zero_grad()
		
		# Forward pass through DNN and CNN models
		cnn_outputs = cnn_model(data)
		
		# Calculate losses for both models
		cnn_loss = criterion(cnn_outputs, targets)
		
		# Backward and optimize for both models
		cnn_loss.backward()
		
		cnn_optimizer.step()
	
	cnn_model.eval()
	with torch.no_grad():
		dnn_test_loss = 0.0
		cnn_test_loss = 0.0
		for data, targets in test_loader:
			cnn_outputs = cnn_model(data)
			cnn_test_loss += criterion(cnn_outputs, targets).item()
	
	print(
		f"Epoch {epoch + 1}/{num_epochs}, DNN Test Loss: {dnn_test_loss / len(test_loader):.4f}, CNN Test Loss: {cnn_test_loss / len(test_loader):.4f}")
