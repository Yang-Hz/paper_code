from datetime import datetime

import pymysql


# 连接池

# 本地数据库
def localConnect():
	connect = pymysql.connect(
		host='localhost',
		port=3306,
		database='grid',
		user='',
		passwd='',
		charset='utf8')
	return connect
