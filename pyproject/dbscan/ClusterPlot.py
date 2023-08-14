import concurrent.futures
import os

import numpy as np
from matplotlib import pyplot as plt

from pyproject.connect.tool.LocalConnect import localConnect


def clusterPlot(user_id, cut_size_select, save_path, cluster_num):
	uid = user_id
	query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC"
	cursor = localConnect().cursor()
	cursor.execute(query, (str(uid),))
	result = cursor.fetchall()
	user_load = [row[2] for row in result]
	loop = int(len(user_load)/cut_size_select)
	for i in range(loop):
		plt.plot(user_load[cut_size_select*i:cut_size_select*(i+1)])
	figure_save_path = save_path
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend([])
	plt.title('CLUSTER:' + str(cluster_num))
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	plt.savefig(os.path.join(figure_save_path, 'CLUSTER_' + str(cluster_num)+'_'+str(user_id)))
	plt.show()
	
	
if __name__ == '__main__':
	random_user_ids = np.random.randint(1000, 7444, 5)
	count = 8
	for user_id in random_user_ids:
		with concurrent.futures.ThreadPoolExecutor(max_workers = 15) as executor:
			executor.submit(
				clusterPlot(user_id, 48, 'D:/paper_result/cluster_result', count)
			)
			executor.submit(
				clusterPlot(user_id, 48 * 7, 'D:/paper_result/cluster_result_week', count)
			)
			executor.submit(
				clusterPlot(user_id, 48 * 7 * 4, 'D:/paper_result/cluster_result_month', count)
			)
		count = count + 1
		if count > 8:
			count = 1