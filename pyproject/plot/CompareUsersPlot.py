import concurrent.futures
import os

import numpy as np
from matplotlib import pyplot as plt

from pyproject.connect.tool.LocalConnect import localConnect


def compareDay(user_count, save_path):
	random_user_ids = np.random.randint(1000, 7444, user_count)
	for user_id in random_user_ids:
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 100"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(user_id),))
		result = cursor.fetchall()
		column_values = [row[2] for row in result]
		plt.plot(column_values[0:48], label="User_id_" + str(result[0][0]))
	figure_save_path = save_path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend()
	# plt.title('compareDay')
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, str(random_user_ids)))
	plt.show()
def compareBigDay(user_count, save_path):
	random_user_ids = np.random.randint(1000, 7444, user_count)
	for user_id in random_user_ids:
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 100"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(user_id),))
		result = cursor.fetchall()
		column_values = [row[2] for row in result]
		plt.plot(column_values[0:48])
	figure_save_path = save_path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend([])
	# plt.title('compareDay')
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, str(len(random_user_ids))+'_'+str(random_user_ids[0])))
	plt.show()

def compareWeek(user_count, save_path):
	random_user_ids = np.random.randint(1000, 7444, user_count)
	for user_id in random_user_ids:
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 228"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(user_id),))
		result = cursor.fetchall()
		column_values = [row[2] for row in result]
		plt.plot(column_values[0:228], label="User_id_" + str(result[0][0]))
	figure_save_path = save_path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend()
	# plt.title('compareDay')
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, str(random_user_ids)))
	plt.show()

def compareMonth(user_count, save_path):
	random_user_ids = np.random.randint(1000, 7444, user_count)
	for user_id in random_user_ids:
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 1344"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(user_id),))
		result = cursor.fetchall()
		column_values = [row[2] for row in result]
		plt.plot(column_values[0:228*4], label="User_id_" + str(result[0][0]))
	figure_save_path = save_path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend()
	# plt.title('compareDay')
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, str(random_user_ids)))
	plt.show()

def compareBigWeek(user_count, save_path):
	random_user_ids = np.random.randint(1000, 7444, user_count)
	for user_id in random_user_ids:
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 228"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(user_id),))
		result = cursor.fetchall()
		column_values = [row[2] for row in result]
		plt.plot(column_values[0:48*7])
	figure_save_path = save_path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend([])
	# plt.title('compareDay')
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, str(len(random_user_ids))+'_'+str(random_user_ids[0])))
	plt.show()

def compareBigMonth(user_count, save_path):
	random_user_ids = np.random.randint(1000, 7444, user_count)
	for user_id in random_user_ids:
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC LIMIT 1344"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(user_id),))
		result = cursor.fetchall()
		column_values = [row[2] for row in result]
		plt.plot(column_values[0:48*7*4])
	figure_save_path = save_path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend([])
	# plt.title('compareDay')
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, str(len(random_user_ids))+'_'+str(random_user_ids[0])))
	plt.show()

if __name__ == '__main__':
	# 不带图标，数大
	# random_user_ids = np.random.randint(20, 100, 150)
	# for i in range(len(random_user_ids)):
	# 	compareBigDay(random_user_ids[i], 'D:/paper_result/compare_user_day')
 	# # 带图标，数小不精致
	# random_user_ids_min = np.random.randint(1, 10, 50)
	# for q in range(len(random_user_ids_min)):
	# 	compareDay(random_user_ids_min[q], 'D:/paper_result/compare_user_day')
# 	对比周
	
		random_user_ids_min = np.random.randint(1, 10, 20)
		
		with concurrent.futures.ThreadPoolExecutor() as executor:
			futures = []
			for q in range(len(random_user_ids_min)):
				futures.append(
					executor.submit(compareDay, random_user_ids_min[q], 'D:/paper_result/compare_user_day'))
				futures.append(
					executor.submit(compareWeek, random_user_ids_min[q], 'D:/paper_result/compare_user_week'))
				futures.append(
					executor.submit(compareMonth, random_user_ids_min[q], 'D:/paper_result/compare_user_Month'))
		
		
		random_user_ids = np.random.randint(30, 150, 20)
		with concurrent.futures.ThreadPoolExecutor() as executor:
			futures = []
			for i in range(len(random_user_ids)):
				futures.append(
					executor.submit(compareBigDay, random_user_ids_min[i], 'D:/paper_result/compare_user_day'))
				futures.append(
					executor.submit(compareBigWeek, random_user_ids_min[i], 'D:/paper_result/compare_user_Week'))
				futures.append(
					executor.submit(compareBigMonth, random_user_ids_min[i], 'D:/paper_result/compare_user_Month'))