import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from char4.MFCLT import train_loader, test_loader

# Assuming input_size, hidden_size, and output_size are defined
input_size = 7
hidden_size = 64
output_size = 48


class ExpertModule(nn.Module):
	def __init__(self, hidden_size, expert_layers=2):
		super(ExpertModule, self).__init__()
		self.expert_layers = nn.ModuleList([
			nn.Linear(hidden_size, hidden_size) for _ in range(expert_layers)
		])
	
	def forward(self, x):
		for layer in self.expert_layers:
			x = layer(x)
		return x


class PrFormer(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers=2, num_experts=4):
		super(PrFormer, self).__init__()
		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(input_size, hidden_size)
		self.transformer_layers = nn.TransformerEncoderLayer(hidden_size, nhead=4)
		self.transformer_encoder = nn.TransformerEncoder(self.transformer_layers, num_layers=num_layers)
		self.expert_modules = nn.ModuleList([
			ExpertModule(hidden_size) for _ in range(num_experts)
		])
		self.fc = nn.Linear(hidden_size * num_experts, output_size)
	
	def forward(self, x):
		x = self.embedding(x)
		x = self.transformer_encoder(x)
		expert_outputs = [expert(x) for expert in self.expert_modules]
		combined_output = torch.cat(expert_outputs, dim=-1)
		out = self.fc(combined_output)
		return out


# Create instances of the models
prformer_model = PrFormer(input_size, hidden_size, output_size)

# Define your optimizer and criterion
prformer_optimizer = optim.Adam(prformer_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Assuming train_loader and test_loader are defined
num_epochs = 200
for epoch in range(num_epochs):
	prformer_model.train()
	for data, targets in train_loader:
		prformer_optimizer.zero_grad()
		prformer_outputs = prformer_model(data)
		prformer_loss = criterion(prformer_outputs, targets)
		prformer_loss.backward()
		prformer_optimizer.step()
	
	prformer_model.eval()
	with torch.no_grad():
		prformer_test_loss = 0.0
		for data, targets in test_loader:
			prformer_outputs = prformer_model(data)
			prformer_test_loss += criterion(prformer_outputs, targets).item()
	
	print(f"Epoch {epoch + 1}/{num_epochs}, PrFormer Test Loss: {prformer_test_loss / len(test_loader):.4f}")
