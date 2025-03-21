import scipy.io as scio
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.framework import graph_util  
from tensorflow.python.platform import gfile  


# 定义网络超参数
learning_rate = 0.0005
training_iters = 200 
test_batch_size = 200
testing_iters = 200
batch_size = 2
display_step = 5
# 定义网络参数
n_input = 256 # 输入的维度
n_classes = 3 # 标签的维度
dropout = 0.8 # Dropout 的概率
epoch_num = 2 #生成batch时的迭代次数

log_dir = './simple_cnn_log'#可视化数据保存地址

# 导入HRRP数据
file_name = './Train_hrrp.mat'
traindata_base =scio.loadmat(file_name)['aa']
# print(data_base.shape)
hrrp = traindata_base[:,3:]
# print(hrrp)
labels = traindata_base[:,0:3]
# print(label)
file_name2 = './Test_hrrp.mat'
testdata_base = scio.loadmat(file_name2)['bb']
test_hrrp = testdata_base[:,3:3+n_input]
test_labels =testdata_base[:,0:3]


#生成batch数据
def get_batch_data(batch_size=batch_size):
    # 从tensor列表中按顺序或随机抽取一个tensor
    input_queue = tf.compat.v1.train.slice_input_producer([hrrp, labels], shuffle=False)
    hrrp_batch, label_batch = tf.compat.v1.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    return hrrp_batch, label_batch

[hrrp_batch, label_batch] = get_batch_data(batch_size=batch_size)

def get_test_data(batch_size=batch_size):
    # 从tensor列表中按顺序或随机抽取一个tensor
    input_queue = tf.compat.v1.train.slice_input_producer([test_hrrp, test_labels], shuffle=False)
    hrrp_test, label_test = tf.compat.v1.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)
    return hrrp_test, label_test

[hrrp_test, label_test] = get_batch_data(batch_size=test_batch_size)


# 占位符输入
with tf.name_scope('inputs'):
    x = tf.compat.v1.placeholder(tf.float32, [None, n_input],name='x_in')
    y = tf.compat.v1.placeholder(tf.float32, [None, n_classes],name='y_in')
    keep_prob = tf.compat.v1.placeholder(tf.float32,name = 'keep_prob')

# 卷积操作
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1,1,1,1], padding='SAME'),b), name=name)

# 最大下采样操作
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1,k,k,1], padding='SAME', name=name)

# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)

# 定义整个网络
def alex_net(_X, _weights, _biases, _dropout):
    # 向量转为矩阵
    _X = tf.reshape(_X, shape=[-1, 1, n_input, 1])

    with tf.name_scope('layer1'):
    # 卷积层
        conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
        # 下采样层
        pool1 = max_pool('pool1', conv1, k=2)
        # 归一化层
        norm1 = norm('norm1', pool1, lsize=4)
        # Dropout
        norm1 = tf.nn.dropout(norm1, _dropout)

    # 卷积
    with tf.name_scope('layer2'):
        conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # 下采样
        pool2 = max_pool('pool2', conv2, k=2)
        # 归一化
        norm2 = norm('norm2', pool2, lsize=4)
        # Dropout
        norm2 = tf.nn.dropout(norm2, _dropout)

    # 卷积
    with tf.name_scope('layer3'):
        conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
        # 下采样
        pool3 = max_pool('pool3', conv3, k=2)
        # 归一化
        norm3 = norm('norm3', pool3, lsize=4)
        # Dropout
        norm3 = tf.nn.dropout(norm3, _dropout)

    # 全连接层，先把特征图转为向量
    with tf.name_scope('fc'):
        dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
        # 全连接层
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # 网络输出层
    with tf.name_scope('outs'):
        out = tf.add(tf.matmul(dense2, _weights['out']), _biases['out'],name='output')
    return out

# 存储所有的网络参数
with tf.name_scope('Weights'):
    weights = {
        'wc1': tf.Variable(tf.compat.v1.random_normal([1, 36, 1, 64])),
        'wc2': tf.Variable(tf.compat.v1.random_normal([1, 36, 64, 128])),
        'wc3': tf.Variable(tf.compat.v1.random_normal([1, 36, 128, 256])),
        'wd1': tf.Variable(tf.compat.v1.random_normal([1*32*256, 1024])),
        'wd2': tf.Variable(tf.compat.v1.random_normal([1024, 1024])),
        'out': tf.Variable(tf.compat.v1.random_normal([1024, n_classes]))
    }
with tf.name_scope('biases'):
    biases = {
        'bc1': tf.Variable(tf.compat.v1.random_normal([64])),
        'bc2': tf.Variable(tf.compat.v1.random_normal([128])),
        'bc3': tf.Variable(tf.compat.v1.random_normal([256])),
        'bd1': tf.Variable(tf.compat.v1.random_normal([1024])),
        'bd2': tf.Variable(tf.compat.v1.random_normal([1024])),
        'out': tf.Variable(tf.compat.v1.random_normal([n_classes]))
    }

# 构建模型
pred = alex_net(x, weights, biases, keep_prob)

# 定义损失函数和学习步骤
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
with tf.name_scope('train'):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量
init = tf.compat.v1.global_variables_initializer()

merged = tf.compat.v1.summary.merge_all()
# train_writer = tf.summary.FileWriter(log_dir+'/train',sess.graph)
# test_writer = tf.summary.FileWriter(log_dir+'/test')

# 开启一个训练
with tf.compat.v1.Session() as sess:
    sess.run(init)
    saver = tf.compat.v1.train.Saver()
    train_writer = tf.compat.v1.summary.FileWriter(log_dir+'/train',sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter(log_dir+'/test')
    step = 1
    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess, coord)
    print('doing1')
    # Keep training until reach max iterations
    try:
        while step * batch_size <= training_iters:
            # print('doing2')
            batch_xs, batch_ys = sess.run([hrrp_batch, label_batch])
            # print(batch_xs.shape)
            # print(batch_ys.shape)
            # 获取批数据
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
            if step % display_step == 0:
                summary= sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                test_writer.add_summary(summary, step)
                # 计算精度
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                # 计算损失值
                loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                
                print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
                # if step % (2*display_step) == 0:
                #     run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                #     run_metadata = tf.RunMetadata()
                #     saver.save(sess,log_dir+'/model.ckpt',step)
            step += 1
        print ("Optimization Finished!")
        step2 = 1
        # 计算测试精度
        while step2 * test_batch_size <= testing_iters:
            test_xs, test_ys = sess.run([hrrp_test, label_test])
            bcc = sess.run(accuracy, feed_dict={x: test_xs, y: test_ys, keep_prob: 1.})
            print ("Iter " + str(step2*test_batch_size) + ", Testing Accuracy = " + "{:.5f}".format(bcc), )
            step2 += 1
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)

    # pb_file_path = 'D:/HRRP data/data/'
    # constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['outs.output'])
    # with tf.gfile.FastGFile(pb_file_path+'model.pb', mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())


 
