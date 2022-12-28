import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图


import csv

csvf = csv.reader(open('./res.csv', 'r'))
data = []
for s in csvf:
  dt = [float(s[0]),float(s[1]),float(s[2]), float(s[3])]
  data.append(dt)

data = np.array(data)

x = data[:, 0]
y = data[:, 1]
succ = data[:, 2]
avgb = data[:, 3]
lavgb = data[:, 3] * data[:, 1]
m = data[:,3]*data[:,1]*data[:,0]
# print(succ)
# avgAB = data[:, 4]
# allB = data[:, 5]


 
# 绘制散点图
"""
ax = Axes3D(fig)
ax.scatter(x, y, avgb)
 
 
# 添加坐标轴(顺序是Z, Y, X)
ax.set_zlabel('SuccessRate(%)', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('INTEREST_N', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('Agent_N', fontdict={'size': 15, 'color': 'red'})
"""

plt.scatter(y, m, s=x*3,c=x)
plt.ylabel('Total Steps for All Target')
plt.xlabel('Target_N')
plt.show()
