import datetime

from pyproject.connect.tool.LocalConnect import localConnect


# 初始化数据库字段信息
# 连接数据库
if __name__ == '__main__':
    total_page = 1000
    page_size = 1000
    cursor = localConnect().cursor()
    for page in range(total_page):
        offset = page * page_size
        sql = "SELECT * FROM loadtest WHERE create_time IS NULL LIMIT %s OFFSET %s"
        cursor.execute(sql, (page_size, offset))
        results = cursor.fetchall()
        sql = "UPDATE loadtest SET create_time = %s WHERE user_id = %s"
        cursor.execute(sql, (datetime.datetime.now(),))
        localConnect().commit()
localConnect().close()


