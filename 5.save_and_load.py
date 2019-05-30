#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np

# settings
LOGDIR='../log/learn/train'
MODELDIR='./models/and'

# data prepare
input_data=tf.constant([[1,1],[1,0],[0,1],[0,0]])
label_data=tf.constant([[1],[0],[0],[0]])
dataset=tf.data.Dataset.from_tensor_slices((input_data,label_data))
dataset=dataset.repeat() # inf data when the params is None
dataset=dataset.shuffle(128)
dataset=dataset.batch(4)
it=dataset.make_one_shot_iterator()
next_batch=it.get_next()

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

writer=tf.summary.FileWriter(LOGDIR,tf.get_default_graph())
saver=tf.train.Saver()

# run graph
with tf.Session() as sess:
    try:
        saver.restore(sess,MODELDIR)
    except:
        sess.run(init)
        for i in range(1000):
            x,y=sess.run(next_batch)
            _,summary=sess.run([optimize,log],feed_dict={inputs:x,labels:y})
            writer.add_summary(summary,i)
        saver.save(sess,MODELDIR)
    # predict
    x,y=sess.run(next_batch)
    predict_value=sess.run(outputs,feed_dict={inputs:x,labels:y})
    print("features    is ")
    print(x)
    print("predict     is ")
    print(predict_value)
    print("groundtruth is ")
    print(y)
