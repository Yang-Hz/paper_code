import random
import matplotlib.pyplot as plt
import numpy as np

from connect.tool.LocalConnect import localConnect


def get_data_by_user_id(user_id):
	query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 4000"
	cursor = localConnect().cursor()
	cursor.execute(query, (str(user_id),))
	result = cursor.fetchall()
	sorted_result = sorted(result, key=lambda x: x[1])
	return sorted_result


def get_data_feature(result):
	feature = []
	
	# Filter out None values before calculating the mean
	numeric_values = [value[2] for value in result if isinstance(value[2], (int, float))]
	
	if numeric_values:
		stamp_db_mean = np.nanmean(numeric_values)
		stame_db_var = np.var(numeric_values)
		stame_db_std = np.std(numeric_values, ddof=1)
		feature.extend([stamp_db_mean, stame_db_var, stame_db_std])
	else:
		feature.extend([0.0, 0.0, 0.0])
	
	return feature


def plot_user_data(result, label, color, linewidth):
	column_values = [row[2] for row in result[:48]]
	plt.plot(column_values, label=label, linewidth=linewidth, color=color)


def get_random_user_id():
	# 获取所有用户ID列表，这里假设用户ID范围在1000到9999之间
	all_user_ids = range(1000, 10000)
	return random.choice(all_user_ids)


def get_data_by_user_id(user_id):
	query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 100"
	cursor = localConnect().cursor()
	cursor.execute(query, (str(user_id),))
	result = cursor.fetchall()
	sorted_result = sorted(result, key=lambda x: x[1])
	return sorted_result


def plot_user_power_consumption(user_id):
	data = get_data_by_user_id(user_id)
	power_consumption = [row[2] for row in data]
	prower_48 = power_consumption[0:48 * 7]
	# Plot the entire day's power consumption
	plt.plot(prower_48, label=f'User {user_id}')


if __name__ == '__main__':
	for i in range(30):
		random_user_ids = random.sample(range(1000, 7444), 7)
		user_data = []
		
		with open('user_ids.txt', 'w') as id_file:
			for user_id in random_user_ids:
				data = get_data_by_user_id(user_id)
				user_data.append((user_id, data))
				# Save user_id to the file
				id_file.write(f"{user_id}\n")
		user_data = sorted(user_data,
		                   key=lambda x: np.nanmean([value[2] for value in x[1] if isinstance(value[2], (int, float))]))
		plt.figure(figsize=(10, 6))
		user_id = []
		for idx, (user_id, data) in enumerate(user_data):
			feature = get_data_feature(data)
			
			color = plt.cm.viridis(idx / len(user_data))
			
			print(user_id)
		
		plt.xlabel('TIME_SLICING/30_Min')
		plt.ylabel('LOAD/MW')
		plt.legend(loc='upper left')
		plt.show()
	
	for i in range(20):
		# 随机选择一个用户ID
		random_user_id = get_random_user_id()
		
		# 获取并绘制该用户的功耗曲线
		plt.figure(figsize=(10, 6))
		plot_user_power_consumption(random_user_id)
		
		plt.xlabel('Timestamp')
		plt.ylabel('Power consumption(MW)')
		plt.title(f'Power Consumption for User {random_user_id}')
		plt.legend()
