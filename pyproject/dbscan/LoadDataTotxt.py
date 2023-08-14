import numpy as np

from pyproject.connect.tool.LocalConnect import localConnect

def WriteTxt(result, filename):
	with open(filename, 'w') as f:
		# 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
		for i in range(len(result)):
			if i > 0 and i % 2 == 1:
				f.write(str(round(result[i], 3)) + '\n')
				continue
			f.write(str(round(result[i], 3))+',')
def WriteTxtK(result, filename):
	with open(filename, 'w') as f:
		# 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
		for i in range(len(result)):
			if isinstance(result[i], int):
				f.write(str(round(result[i], 3)) + '\n')
				continue
			f.write(str(round(result[i], 3))+',')
			
		
if __name__ == '__main__':
	
	random_integers = np.random.randint(1000, 7444, 100)
	# txt_input = []
	# for uid in random_integers:
	# 	query = "SELECT * FROM loadtest LEFT JOIN role ON loadtest.user_id = role.user_id WHERE loadtest.user_id = %s ORDER BY data DESC LIMIT 8"
	# 	cursor = localConnect().cursor()
	# 	cursor.execute(query, (str(uid),))
	# 	result = cursor.fetchall()
	# 	if result[0][6] != 1:
	# 		continue
	#
	# 	user_load = [row[2] for row in result]
	# 	for i in range(8):
	# 		txt_input.append(user_load[i])
	# 	txt_input.append(np.mean(user_load))
	# 	txt_input.append(np.var(user_load))
	# 	txt_input.append(np.std(user_load))
	# 	txt_input.append(1)
		
	# WriteTxtK(txt_input, 'Users_300_1')
	# txt_input_2 = []

	# for uid in random_integers:
	# 	query = "SELECT * FROM loadtest LEFT JOIN role ON loadtest.user_id = role.user_id WHERE loadtest.user_id = %s ORDER BY data DESC LIMIT 8"
	# 	cursor = localConnect().cursor()
	# 	cursor.execute(query, (str(uid),))
	# 	result = cursor.fetchall()
	#
	# 	if len(result) > 6 and result[0][6] != 2:
	# 		continue
	# 	user_load = [row[2] for row in result]
	# 	for i in range(8):
	# 		txt_input_2.append(user_load[i])
	# 	txt_input_2.append(np.mean(user_load))
	# 	txt_input_2.append(np.var(user_load))
	# 	txt_input_2.append(np.std(user_load))
	# 	txt_input_2.append(2)
	# 	if np.array(txt_input_2).shape[0] > 60 * 12:
	#
	# 		break
	#
	# WriteTxtK(txt_input_2, 'Users_300_2')
	load_input = []
	
	for uid in random_integers:
		query = "SELECT * FROM loadtest LEFT JOIN role ON loadtest.user_id = role.user_id WHERE loadtest.user_id = %s ORDER BY data DESC LIMIT 8"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(uid),))
		result = cursor.fetchall()
		user_load = [row[2] for row in result]
		load_input.append(np.var(user_load))
		load_input.append(np.std(user_load))
	WriteTxt(load_input, 'User_1000')
