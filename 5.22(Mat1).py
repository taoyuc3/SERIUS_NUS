from matplotlib import pyplot as plt
x =  [2,5,8,10,14]
y =  [17,12,16,6,9]
x2 =  [3,6,9,11,16]
y2 =  [30,6,15,7,3]
x3 =  [1,7]
y3 =  [9,7]
x4 = [4,12,15]
y4 = [8,8,19]
x5 = [13,17]
y5 = [2,5.5]
plt.bar(x, y, align =  'center')
plt.bar(x2, y2, color =  'darkgreen', align =  'center')
plt.bar(x3, y3, color =  'orange', align =  'center')
plt.bar(x4, y4, color =  'y', align =  'center')
plt.bar(x5, y5, color =  'red', align =  'center')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()