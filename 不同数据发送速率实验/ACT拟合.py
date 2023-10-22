import numpy as np
import matplotlib.pyplot as plt

# 五组实验arrival rate
x = [2, 4, 6, 8, 10, 12] 

# 每组对应的平均JCT  
y = [7.77, 7.90, 7.81, 7.86, 7.79, 7.73]

fig, ax = plt.subplots()

# 绘制数据点  
ax.plot(x, y, 'o') 

# 画图
ax.plot(x, y)
plt.xlabel('Job Arrival Rate')  
plt.ylabel('Average JCT (s)')
plt.title('Average JCT with Different Job Arrival Rate')

ax.set_xlim(0, max(x) + 2)
ax.set_ylim(7, max(y) + 1)

plt.show()