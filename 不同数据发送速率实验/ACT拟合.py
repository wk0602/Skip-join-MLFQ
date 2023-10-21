import numpy as np
import matplotlib.pyplot as plt

# 五组实验arrival rate
x = [5, 10, 15, 20, 25] 

# 每组对应的平均JCT  
y = [17.07, 22.37, 24.50, 25.36, 26.20]

fig, ax = plt.subplots()

# 绘制数据点  
ax.plot(x, y, 'o') 

# 画图
ax.plot(x, y)
plt.xlabel('Job Arrival Rate')  
plt.ylabel('Average JCT (s)')
plt.title('Average JCT with Different Job Arrival Rate')

ax.set_xlim(0, max(x) + 2)
ax.set_ylim(10, max(y) + 2)

plt.show()