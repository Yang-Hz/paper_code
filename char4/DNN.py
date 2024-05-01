import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from char4.MFCLT import train_loader, test_loader

# Assuming input_size, hidden_size, and output_size are defined
input_size = 7
hidden_size = 64
output_size = 48


class SimpleDNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(SimpleDNN, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, output_size)
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x


model = SimpleDNN(input_size, hidden_size, output_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 200
for epoch in range(num_epochs):
	model.train()
	for data, targets in train_loader:
		optimizer.zero_grad()
		outputs = model(data)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()
	
	model.eval()
	with torch.no_grad():
		test_loss = 0.0
		for data, targets in test_loader:
			outputs = model(data)
			test_loss += criterion(outputs, targets).item()
	
	print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss / len(test_loader):.4f}")
