import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class CustomCNN(nn.Module):
	def __init__(self):
		super(CustomCNN, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout1 = nn.Dropout(p=0.25)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dropout2 = nn.Dropout(p=0.25)
		self.conv_output_size = self._calculate_conv_output_size()
	
	def _calculate_conv_output_size(self):
		x = torch.randn(1, 1, 64, 48)  # Input tensor shape: (batch_size, channels, height, width)
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		x = self.dropout1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		x = self.dropout2(x)
		return x.size(1) * x.size(2) * x.size(3)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		x = self.dropout1(x)
		
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)
		x = self.dropout2(x)
		x = x.view(-1, self.conv_output_size)
		
		return x


# Data loading and processing
def load_data(filename, batch_size=32, train_ratio=0.8):
	# Create dataset and DataLoader
	dataset = TensorDataset(x, y)
	train_size = int(train_ratio * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
	
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=batch_size)
	
	return train_loader, test_loader


def export_model(model, filename):
	# Export model parameters to a file
	torch.save(model.state_dict(), filename)
	print(f"Model exported to {filename}")
