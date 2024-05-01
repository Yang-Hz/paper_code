from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn


class DataLoaderBuilder:
	def __init__(self, dataset, batch_size=32, train_ratio=0.8):
		self.dataset = dataset
		self.batch_size = batch_size
		self.train_ratio = train_ratio
		self.train_loader = None
		self.test_loader = None
	
	def build_loaders(self):
		train_size = int(self.train_ratio * len(self.dataset))
		test_size = len(self.dataset) - train_size
		train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
		
		self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
		self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)


class ModelTrainer:
	def __init__(self, model, criterion, optimizer, train_loader, test_loader, num_epochs=10, save_every=1):
		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.num_epochs = num_epochs
		self.save_every = save_every
	
	def train(self):
		for epoch in range(self.num_epochs):
			self.model.train()
			total_train_loss = 0.0
			for batch_features, batch_labels in self.train_loader:
				outputs = self.model(batch_features)
				
				loss = self.criterion(outputs, batch_labels)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				
				total_train_loss += loss.item() * len(batch_features)
			
			average_train_loss = total_train_loss / len(self.train_loader.dataset)
			print(f"Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {average_train_loss:.4f}")
			
			if (epoch + 1) % self.save_every == 0:
				torch.save(self.model.state_dict(), f"model_epoch_{epoch + 1}.pt")
			
			self.evaluate(epoch)
	
	def evaluate(self, epoch):
		self.model.eval()
		total_test_loss = 0.0
		with torch.no_grad():
			for batch_features, batch_labels in self.test_loader:
				outputs = self.model(batch_features)
				loss = self.criterion(outputs, batch_labels)
				total_test_loss += loss.item() * len(batch_features)
		
		average_test_loss = total_test_loss / len(self.test_loader.dataset)
		print(f"Epoch {epoch + 1}/{self.num_epochs}, Test Loss: {average_test_loss:.4f}")


class CustomDatasetBuilder:
	def __init__(self, data_file, seq_length):
		self.data_file = data_file
		self.seq_length = seq_length
	
	def build_dataset(self):
		with open(self.data_file, 'r', encoding='utf-8') as fp:
			lines = fp.readlines()
		
		data = [float(line.replace('\n', '').split(' ')[-1]) for line in lines]
		data = [data[i:i + self.seq_length] for i in range(len(data) - self.seq_length)]
		x = [data[i:i + 7] for i in range(len(data) - 7)]
		y = [data[i + 7] for i in range(len(data) - 7)]
		
		x = torch.Tensor(x)
		y = torch.Tensor(y)
		
		return TensorDataset(x, y)


class ModelBuilder:
	def __init__(self, input_size, hidden_size, output_size, num_layers):
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers
	
	def build_model(self):
		return ComplexModel(self.input_size, self.hidden_size, self.output_size, self.num_layers)


class ComplexModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers):
		super(ComplexModel, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool1d(kernel_size=2)
		self.dropout = nn.Dropout(p=0.2)
		self.lstm1 = nn.LSTM(input_size=12, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)
		self.attention = lfam(7)
	
	def forward(self, x):
		x = self.attention(x)
		x = x.squeeze()
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.dropout(x)
		
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.dropout(x)
		
		x, _ = self.lstm1(x)
		x, _ = self.lstm2(x)
		x = self.fc(x[:, -1, :])  # Use the last LSTM output for prediction
		return x


class lfam(nn.Module):
	def __init__(self, channel, ratio=4, kernel_size=7):
		super(lfam, self).__init__()
		self.channel_attention = channel_attention(channel, ratio)
		self.spacial_attention = spacial_attention(kernel_size)
	
	def forward(self, x):
		x = self.channel_attention(x)
		x = self.spacial_attention(x)
		return x


class channel_attention(nn.Module):
	def __init__(self, channel, ratio):
		super(channel_attention, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // ratio, False),
			nn.ReLU(),
			nn.Linear(channel // ratio, channel, False)
		)
		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		x = x.unsqueeze(-1)
		
		b, c, w, _ = x.size()
		avg_x = self.avg_pool(x).view(b, c)
		max_x = self.max_pool(x).view(b, c)
		avg_x_c = self.fc(avg_x)
		max_x_c = self.fc(max_x)
		
		out = self.sigmoid(avg_x_c + max_x_c).view([b, c, 1]).expand(-1, -1, 48)
		x = x.squeeze()
		return out * x


class spacial_attention(nn.Module):
	def __init__(self, kernel_size=7):
		super(spacial_attention, self).__init__()
		padding = 7 // 2
		self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding, bias=False)
		self.sigmoid = nn.Sigmoid()
