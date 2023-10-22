import matplotlib.pyplot as plt

# 实验数据  
queue_nums = [3,4,5]
queue_rates = [2,4,8,16]
ACT_3 = [5.14, 6.53, 7.85, 9.01] 
ACT_4 = [5.71, 7.16, 9.50, 9.89]
ACT_5 = [6.51, 8.92, 9.91, 9.85]

# 创建图形
fig, ax = plt.subplots() 

# 绘制不同队列数量曲线
ax.plot(queue_rates, ACT_3, '-o', color='red', label='Queue Num=3')  
ax.plot(queue_rates, ACT_4, '-o', color='blue', label='Queue Num=4')
ax.plot(queue_rates, ACT_5, '-o', color='green', label='Queue Num=5')

# 添加横纵坐标标签  
ax.set_xlabel('Queue Rate')
ax.set_ylabel('Average JCT (s)')

# 添加各曲线的图例
ax.legend()

ax.set_xlim(0, 18)
ax.set_ylim(5, 12)
plt.show()