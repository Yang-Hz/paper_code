import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, random_split, DataLoader
import torch
from torchsummary import summary

with open('.txt', 'r', encoding='utf-8') as fp:
	lines = fp.readlines()
i = 0
data = []
for line in lines:
	i += 1
	if i > 2000:
		break
	data.append(float(line.replace('\n', '').split(' ')[-1]))

plt.plot(range(len(data)), data)
plt.show()
data = [data[i:i + 48] for i in range(len(data) - 48)]
x = []
y = []
for i in range(len(data) - 7):
	x.append(data[i:i + 7])
	y.append(data[i + 7])
x = torch.Tensor(x)
y = torch.Tensor(y)

dataset = TensorDataset(x, y)
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


input_size = 7
hidden_size = 64
output_size = 48
num_layers = 2
seq_length = 7
batch_size = 20


class ComplexModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers):
		super(ComplexModel, self).__init__()
		self.lstm1 = nn.LSTM(input_size=output_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(in_channels=64, out_channels=7, kernel_size=3, padding=1)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool1d(kernel_size=2)
		self.dropout = nn.Dropout(p=0.2)
		
		self.fc = nn.Linear(hidden_size + 12, output_size)
	
		
		# Transformer Encoder
		self.d_model = hidden_size + 12
		self.position_embeddings = nn.Embedding(7, self.d_model)
		self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size + 12, nhead=2)
		self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
	
	def forward(self, x):
		lstm_input = x.clone()
		lstm_output, _ = self.lstm1(lstm_input)
		lstm_output, _ = self.lstm2(lstm_output)
		cnn_input = x.clone()
		cnn_output = self.conv1(cnn_input)
		cnn_output = self.relu(cnn_output)
		cnn_output = self.maxpool(cnn_output)
		cnn_output = self.dropout(cnn_output)
		cnn_output = self.conv2(cnn_output)
		cnn_output = self.relu(cnn_output)
		cnn_output = self.maxpool(cnn_output)
		cnn_output = self.dropout(cnn_output)
		
		
		combined_output = torch.cat((lstm_output, cnn_output), dim=2)
		seq_length = combined_output.size(1)
		position_ids = torch.arange(seq_length, dtype=torch.long, device=combined_output.device).unsqueeze(0).expand_as(
			combined_output[:, :, 0])
		position_embeddings = self.position_embeddings(position_ids)
		embeddings = combined_output + position_embeddings
		transformer_output = self.transformer_encoder(embeddings)
		output = self.fc(transformer_output[:, -1, :])
		
		return output


model = ComplexModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers)

model = ComplexModel(input_size, hidden_size, output_size, num_layers)
x = torch.randn(1, 7, 48)
y = model(x)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
summary(model, (1, 7, 48))
num_epochs = 200
warmup_epochs = 3
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: min(epoch / warmup_epochs, 1))

train_epoch_losses = []
test_epoch_losses = []
for epoch in range(num_epochs):
	model.train()
	train_loss = 0
	for batch_idx, (data, targets) in enumerate(train_loader):
		batch_loss = 0
		data = data
		targets = targets
		outputs = model(data)
		loss = criterion(outputs, targets)
		
		optimizer.zero_grad()
		loss.backward()
		train_loss += loss.item()
		optimizer.step()
	
	train_loss /= len(train_loader)
	train_epoch_losses.append(train_loss)
	scheduler.step()
	
	model.eval()
	test_loss = 0.0
	with torch.no_grad():
		for data, targets in test_loader:
			data = data
			targets = targets
			outputs = model(data)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
	
	test_loss /= len(test_loader)
	print('Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, num_epochs, test_loss))
	test_epoch_losses.append(test_loss)
