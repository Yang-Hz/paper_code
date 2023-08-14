import concurrent.futures
import logging
import os

import numpy as np
from matplotlib import pyplot as plt

from pyproject.connect.tool.LocalConnect import localConnect


def Predict(result, title, cut_size, SVM_Val, D_SVM_val):
	user_load = [row[2] for row in result]
	loop = len(user_load)/ cut_size
	user_real = user_load[0: cut_size]
	SVM_index = 0
	is_plot = False
	for i in range(int(loop)):
		if i == 0:
			continue
		if SVM_index == 0:
			SVM_predict = user_load[cut_size * i : cut_size * (i+1)]
			stamp = np.array(SVM_predict) - np.array(user_real)
			stamp_mean = np.mean(stamp)
			stame_var = np.var(stamp)
			stame_std = np.std(stamp, ddof=1)
			if stamp_mean > SVM_Val[0] and stame_var > SVM_Val[1] and stame_std > SVM_Val[2]:
				SVM_index = i
				continue
			else:
				continue
		D_SVM_predict = user_load[cut_size * i : cut_size * (i+1)]
		stamp_db = np.array(D_SVM_predict) - np.array(user_real)
		stamp_db_mean = np.mean(stamp_db)
		stame_db_var = np.var(stamp_db)
		stame_db_std = np.std(stamp_db, ddof=1)
		
		if SVM_index > 0 and stamp_db_mean > D_SVM_val[0] and stame_db_var > D_SVM_val[1] and stame_db_std > D_SVM_val[2]:
			SVM_predict = user_load[cut_size * SVM_index: cut_size * (SVM_index + 1)]
			plt.plot(user_real, label = 'user_real')
			plt.plot(SVM_predict, label = title + '_predict')
			plt.plot(D_SVM_predict, label = 'D_'+title+'_predict')
			figure_save_path = 'D:/paper_result/Predict'+str(title)
			# 如果不存在目录figure_save_path，则创建
			if not os.path.exists(figure_save_path):
				os.makedirs(figure_save_path)
			plt.legend()
			plt.title(str(title))
			plt.xlabel('TIME_SLICING/0.5H')
			plt.ylabel('LOAD/MW')
			# 第一个是指存储路径，第二个是图片名字
			plt.savefig(os.path.join(figure_save_path, 'user_id' + str(result[0][0])+str(i)))
			plt.show()
			is_plot = True
			continue
	if is_plot == False:
		logging.error('FIND EXCEPTION')

if __name__ == '__main__':
	random_integers = np.random.randint(1000, 7444, 5)
	# cut_size = 48 * 7
	cut_size = 48
	# cut_size = 48 * 7 * 4
	for uid in random_integers:
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(uid),))
		result = cursor.fetchall()
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
			executor.submit(
				Predict(result, 'SVM', cut_size=cut_size, SVM_Val=[-0.3, 0, 0], D_SVM_val=[-0.2, 0, 0])
			)
			executor.submit(
				Predict(result, 'CNN', cut_size=cut_size, SVM_Val=[-0.2, 0, 0], D_SVM_val=[-0.15, 0, 0])
			)
			executor.submit(
				Predict(result, 'LSTM', cut_size=cut_size, SVM_Val=[-0.2, 0, 0], D_SVM_val=[-0.15, 0, 0])
			)
			executor.submit(
				Predict(result, 'CNN-LSTM', cut_size=cut_size, SVM_Val=[-0.12, 0, 0], D_SVM_val=[-0.08, 0, 0])
			)
		print(uid)