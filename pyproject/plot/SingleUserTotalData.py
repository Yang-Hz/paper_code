import os

import numpy as np
from matplotlib import pyplot as plt

from pyproject.connect.tool.LocalConnect import localConnect
def singleUser(user_id):
	uid = user_id
	query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC"
	cursor = localConnect().cursor()
	cursor.execute(query, (str(uid),))
	result = cursor.fetchall()
	return result

def plotSingleUser(user_id, save_path):
	user_load = [row[2] for row in singleUser(user_id)]
	plt.plot(user_load, label="User_" + str(user_id))
	plt.legend()
	figure_save_path = save_path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.title('USER_ID:' + str(user_id))
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	plt.savefig(os.path.join(figure_save_path, 'month_user_id' + str(user_id)))
	plt.show()


if __name__ == '__main__':

	random_integers = np.random.randint(1000, 7444, 4)

	load_1 = [row[2] for row in singleUser(random_integers[0])]
	load_2 = [row[2] for row in singleUser(random_integers[1])]
	load_3 = [row[2] for row in singleUser(random_integers[2])]
	load_4 = [row[2] for row in singleUser(random_integers[3])]
	
	plt.plot(load_1, label="User_" + str(random_integers[0]))
	plt.plot(load_2, label="User_" + str(random_integers[1]))
	plt.plot(load_3, label="User_" + str(random_integers[2]))
	plt.plot(load_4, label="User_" + str(random_integers[3]))
	
	plt.legend()
	plt.title('USER_ID:' + str(singleUser(random_integers[0])[0][0])+'&'+str(singleUser(random_integers[1])[0][0])
	          +'&'+str(singleUser(random_integers[2])[0][0])+'&'+str(singleUser(random_integers[3])[0][0]))
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	plt.show()

