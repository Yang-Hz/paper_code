import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class CustomLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers=1):
		super(CustomLSTM, self).__init__()
		
		self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.output_length = 64
		self.total_params = self._calculate_total_params()
	
	def _calculate_total_params(self):
		params_lstm1 = sum(p.numel() for p in self.lstm1.parameters())
		params_lstm2 = sum(p.numel() for p in self.lstm2.parameters())
		return params_lstm1 + params_lstm2
	
	def forward(self, x):
		out, _ = self.lstm1(x)
		out, _ = self.lstm2(out)
		out = out[:, -self.output_length:, :]
		
		return out


# Data loading and processing
def load_data(filename, batch_size=32, train_ratio=0.8):
	x = []
	y = []
	dataset = TensorDataset(x, y)
	train_size = int(train_ratio * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
	
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size)
	
	return train_loader, test_loader


def export_model(model, filename):
	torch.save(model.state_dict(), filename)
	print(f"Model exported to {filename}")
