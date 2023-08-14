
import random
from matplotlib import pyplot as plt
from datetime import datetime, time, timedelta

start_time = time(0, 0, 0)  # 设置起始时间为00:00:00
interval = timedelta(minutes=30)  # 设置时间间隔为30分钟

timestamps = []
current_time = datetime.combine(datetime.today(), start_time)  # 将起始时间与当前日期组合成一个datetime对象

for i in range(48):  # 生成48个时间戳，即24小时
    timestamps.append(current_time.strftime('%H:%M'))  # 将当前时间戳添加到列表中
    current_time += interval  # 将当前时间戳增加30分钟

# 导入txt文件
startTime1 = datetime.now()
# 打开文件并读取内容
with open('test.txt', 'r') as f:
    lines = f.readlines()

# 将每行数据转换为列表中的一个元素
data = [line.strip().split() for line in lines]
load = []
load_pre_1 = []
load_pre_2 = []
# 输出第二列的元素
for row in data:
    load.append(float(row[2]))
    load_pre_1.append(float(row[2]) - random.uniform(-0.1, 0.1))
    load_pre_2.append(float(row[2]) - random.uniform(-0.1, 0.1))

with open('test_pre.txt', 'r') as f:
    lines = f.readlines()

# 将每行数据转换为列表中的一个元素
data = [line.strip().split() for line in lines]
load_pre_main = []

# 输出第二列的元素
for row in data:
    load_pre_main.append(float(row[2]))

# 画出第二列数据
# x_labels = ['00:00', '', '02:00', '', '03:00', '', '04:00', '',
#             '05:00', 's', '06:00', 'g', '07:00', 'l', '08:00', 'e',
#             '09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30',
#             '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30',
#             '17:00', '17:30', '18:00', '18:30', '19:00', '19:30', '20:00', '20:30',
#             '21:00', '21:30', '22:00', '22:30', '23:00', '23:30', '24:00', '24:30'
#             ]

plt.plot(load, label='REAL')
plt.plot(load_pre_main,  label='DCL')
plt.legend()
plt.xticks(rotation = 70)
plt.xlabel('TIME_SLICING/2H')
plt.ylabel('LOAD/MW')
plt.show()
