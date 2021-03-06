#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np

# data prepare
input_data=[[1,1],[1,0],[0,1],[0,0]]
label_data=[[1],[0],[0],[0]]

# build graph with default
inputs=tf.placeholder(tf.float64,shape=[None,2])
weights=tf.Variable(np.random.uniform(size=[2,1]))
bias=tf.Variable(np.random.uniform(size=[1,]))
outputs=tf.matmul(inputs,weights)+bias
labels=tf.placeholder(tf.float64,shape=[None,1])
loss=tf.reduce_mean(tf.square(labels-outputs))
optimize=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init=tf.global_variables_initializer()

# tensorboard summary
tf.summary.scalar('loss',loss)
log=tf.summary.merge_all()

writer=tf.summary.FileWriter('../log/tmp/train',tf.get_default_graph())


# run graph
with tf.Session() as sess:
    sess.run(init)
    # train 1000 epcho
    for i in range(1000):
        _,summary=sess.run([optimize,log],feed_dict={inputs:input_data,labels:label_data})
        # write log
        writer.add_summary(summary,i)

    # predict
    predict_value=sess.run(outputs,feed_dict={inputs:input_data,labels:label_data})
    print("predict     is ")
    print(predict_value)
    print("groundtruth is ")
    print(label_data)
