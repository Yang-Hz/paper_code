import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from char4.MFCLT import test_loader, train_loader

# Assuming input_size, hidden_size, and output_size are defined
input_size = 7
hidden_size = 64
output_size = 48


class ReversibleLayer(nn.Module):
	def __init__(self, hidden_size, num_heads=4, feed_forward_dim=256, dropout=0.2):
		super(ReversibleLayer, self).__init__()
		self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
		self.feed_forward = nn.Sequential(
			nn.Linear(hidden_size, feed_forward_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(feed_forward_dim, hidden_size),
		)
	
	def forward(self, x):
		residual = x
		x = self.attention(x, x, x)[0] + residual
		x = self.feed_forward(x) + x
		return x


class FlowFormer(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_heads=4, feed_forward_dim=256,
	             dropout=0.2):
		super(FlowFormer, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.reversible_layers = nn.ModuleList([
			ReversibleLayer(hidden_size, num_heads, feed_forward_dim, dropout) for _ in range(num_layers)
		])
		self.fc = nn.Linear(hidden_size, output_size)
	
	def forward(self, x):
		x = self.embedding(x)
		for layer in self.reversible_layers:
			x = layer(x)
		out = self.fc(x)
		return out


# Create instances of the models
flowformer_model = FlowFormer(input_size, hidden_size, output_size)

# Define your optimizer and criterion
flowformer_optimizer = optim.Adam(flowformer_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Assuming train_loader and test_loader are defined
num_epochs = 200
for epoch in range(num_epochs):
	flowformer_model.train()
	for data, targets in train_loader:
		flowformer_optimizer.zero_grad()
		flowformer_outputs = flowformer_model(data)
		flowformer_loss = criterion(flowformer_outputs, targets)
		flowformer_loss.backward()
		flowformer_optimizer.step()
	
	flowformer_model.eval()
	with torch.no_grad():
		flowformer_test_loss = 0.0
		for data, targets in test_loader:
			flowformer_outputs = flowformer_model(data)
			flowformer_test_loss += criterion(flowformer_outputs, targets).item()
	
	print(f"Epoch {epoch + 1}/{num_epochs}, FlowFormer Test Loss: {flowformer_test_loss / len(test_loader):.4f}")
