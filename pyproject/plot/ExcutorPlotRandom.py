import concurrent

import numpy as np

from DayWeekMonth import day, week, month
from pyproject.connect.tool.LocalConnect import localConnect

# 均匀抽样 从全量用户中抽取指定数量用户进行可视化
if __name__ == '__main__':
	# 数据库索引上限7444
	# 在索引范围内随机抽样200用户数据
	random_integers = np.random.randint(1000, 7444, 200)
	for uid in random_integers:
		query = "SELECT * FROM loadtest WHERE user_id = %s ORDER BY data DESC"
		cursor = localConnect().cursor()
		cursor.execute(query, (str(uid),))
		result = cursor.fetchall()
		if len(result) == 0:
			continue
		column_values = [row[2] for row in result]
		with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
			for i in range(5):
				executor.submit(day(column_values, 48, result, 'D:/paper_result/200_random_day'))
				executor.submit(week(column_values, 48 * 7, result, 'D:/paper_result/200_random_week'))
				executor.submit(month(column_values, 48 * 7 * 4, result, 'D:/paper_result/200_random_month'))