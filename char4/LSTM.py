import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from char4.MFCLT import test_loader, train_loader

# Assuming input_size, hidden_size, and output_size are defined
input_size = 7
hidden_size = 64
output_size = 48


class SimpleLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(SimpleLSTM, self).__init__()
		self.hidden_size = hidden_size
		self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)
	
	def forward(self, x):
		h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
		c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
		out, _ = self.lstm(x, (h0, c0))
		out = self.fc(out[:, -1, :])
		return out


# Create instances of the models
lstm_model = SimpleLSTM(input_size, hidden_size, output_size)

# Define your optimizer and criterion
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Assuming train_loader and test_loader are defined
num_epochs = 200
for epoch in range(num_epochs):
	lstm_model.train()
	for data, targets in train_loader:
		lstm_optimizer.zero_grad()
		lstm_outputs = lstm_model(data)
		lstm_loss = criterion(lstm_outputs, targets)
		lstm_loss.backward()
		lstm_optimizer.step()
	
	lstm_model.eval()
	with torch.no_grad():
		lstm_test_loss = 0.0
		for data, targets in test_loader:
			lstm_outputs = lstm_model(data)
			lstm_test_loss += criterion(lstm_outputs, targets).item()
	
	print(f"Epoch {epoch + 1}/{num_epochs}, LSTM Test Loss: {lstm_test_loss / len(test_loader):.4f}")
