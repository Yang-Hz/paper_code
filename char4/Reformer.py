import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from char4.MFCLT import test_loader, train_loader

# Assuming input_size, hidden_size, and output_size are defined
input_size = 7
hidden_size = 64
output_size = 48


class Reformer(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Reformer, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = 2  # Number of Reformer layers
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.reformer_layers = nn.ModuleList([
			nn.ReformerLayer(
				hidden_size=hidden_size,
				num_heads=4,
				feed_forward_dim=256,
				dropout=0.2,
				layer_dropout=0.1,
				lsh_dropout=0.1
			) for _ in range(self.num_layers)
		])
		self.fc = nn.Linear(hidden_size, output_size)
	
	def forward(self, x):
		x = self.embedding(x)
		for layer in self.reformer_layers:
			x = layer(x)
		out = self.fc(x)
		return out


# Create instances of the models
reformer_model = Reformer(input_size, hidden_size, output_size)

# Define your optimizer and criterion
reformer_optimizer = optim.Adam(reformer_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Assuming train_loader and test_loader are defined
num_epochs = 200
for epoch in range(num_epochs):
	reformer_model.train()
	for data, targets in train_loader:
		reformer_optimizer.zero_grad()
		reformer_outputs = reformer_model(data)
		reformer_loss = criterion(reformer_outputs, targets)
		reformer_loss.backward()
		reformer_optimizer.step()
	
	reformer_model.eval()
	with torch.no_grad():
		reformer_test_loss = 0.0
		for data, targets in test_loader:
			reformer_outputs = reformer_model(data)
			reformer_test_loss += criterion(reformer_outputs, targets).item()
	
	print(f"Epoch {epoch + 1}/{num_epochs}, Reformer Test Loss: {reformer_test_loss / len(test_loader):.4f}")
