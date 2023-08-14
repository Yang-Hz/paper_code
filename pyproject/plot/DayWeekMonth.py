import concurrent.futures
import os

import pymysql
from matplotlib import pyplot as plt

# 以周为单位进行可视化
def week(column_values, print_select, result, path):
	for i in range(8):
		plt.plot(column_values[print_select * i: print_select * (i + 1)], label="week_" + str(i + 1))
	figure_save_path = path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend()
	plt.title('USER_ID:' + str(result[0][0]))
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, 'week_user_id' + str(result[0][0])))
	plt.show()

# 以月为单位进行可视化
def month(column_values, print_select, result, path):
	for i in range(4):
		plt.plot(column_values[print_select * i: print_select * (i + 1)], label="month_" + str(i + 1))
	figure_save_path = path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend()
	plt.title('USER_ID:' + str(result[0][0]))
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, 'month_user_id' + str(result[0][0])))
	plt.show()

# 以天为单位进行可视化
def day(column_values, print_select, result, path):
	for i in range(7):
		plt.plot(column_values[print_select * i: print_select * (i + 1)], label="day_" + str(i + 1))
	figure_save_path = path
	# 如果不存在目录figure_save_path，则创建
	if not os.path.exists(figure_save_path):
		os.makedirs(figure_save_path)
	plt.legend()
	plt.title('USER_ID:' + str(result[0][0]))
	plt.xlabel('TIME_SLICING/0.5H')
	plt.ylabel('LOAD/MW')
	# 第一个是指存储路径，第二个是图片名字
	plt.savefig(os.path.join(figure_save_path, 'day_user_id' + str(result[0][0])))
	plt.show()
