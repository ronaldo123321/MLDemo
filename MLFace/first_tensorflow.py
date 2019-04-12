import tensorflow as tf
import numpy as np
# #----------------------tensorflow基本输入结构的定义区------------------------------------------
# #在tensorflow中，所有变量必须用tf.Variable定义，0为初始值，name为该变量名称（可选）
# state = tf.Variable(0,name='counter')
# print(state.name)
# one = tf.constant(1)
#
# #与new_value=state+1等价，tf中必须用专有函数操作，且下述语句仅仅是定义该操作，并不执行
# new_value = tf.add(state,one)
# #指定将new_value的值更新到state，这里依然是事先指定这个操作给变量update.并不会执行
# update = tf.assign(state,new_value)
#
# #指定tf启动session时要进行所有变量初始化操作，这里依然只是指定初始化给init,并不实际
# # 执行，换句话说，只要上面存在tf.Variable（）的调用，就必须调用init操作
# init = tf.global_variables_initializer()
# #----------------------tensorflow基本输入结构的定义区------------------------------------------
#
# #启动tf会话并执行执行上面定义的操作
# with tf.Session() as sess:
#     sess.run(init)  #所有tf.Vaiable()定义的变量被真正初始化
#     print('-' * 8,sess.run(state))
#     #测试变量更新操作，执行多次
#     for _ in range(10):
#         sess.run(update)
#         print(sess.run(state))



#利用Numpy生成测试数据，共100个点
x_data = np.float32(np.random.rand(2,100))
#np.dot  矩阵标准乘运算
y_data = np.dot([0.100,0.700],x_data) + 0.300 #等价于线性方程：y=0.1 * x2 + 0.7 * x2 + 0.3
print(x_data)
print(y_data)

#构造一个线性模型
b = tf.Variable(tf.zeros(1))     #偏移系数初始为0
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))   #产生一个1行2列的矩阵，数值从-1到1之间，为权重的初始值
y = tf.matmul(W,x_data) + b     #产生训练数据

#最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))     #与真实数据y_data之间的方差
optimizer = tf.train.GradientDescentOptimizer(0.5)  #梯度下降步长
train = optimizer.minimize(loss)        #使用梯度下降算法开始优化数据使其变到最小

#初始化变量，tf中只要定义变量就必须初始化变量
init = tf.global_variables_initializer()

#启动图
sess = tf.Session()
sess.run(init)   #执行变量初始化

#拟合平面，201此循环执行sess.run(train)
for step in range(0,401):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(W),sess.run(b))

