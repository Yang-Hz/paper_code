import concurrent


from DayWeekMonth import day, week, month
from pyproject.connect.tool.LocalConnect import localConnect
# 全量将所有用户的功耗数据可视化
if __name__ == '__main__':
	
	# 数据库索引上限7444
	for uid in range(7445):
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(uid),))
		result = cursor.fetchall()
		if len(result) == 0:
			continue
		column_values = [row[2] for row in result]
		# 线程池同步画 day\week\month图
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
			for i in range(5):
				executor.submit(day(column_values, 48, result, 'D:/paper_result/file_fig_day'))
				executor.submit(week(column_values, 48 * 7, result, 'D:/paper_result/file_fig_week'))
				executor.submit(month(column_values, 48 * 7 * 4, result, 'D:/paper_result/file_fig_month'))
