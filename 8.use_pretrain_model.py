#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import sys


# settings
LOGDIR='../log/learn/train'
# ./models must be created!!!
MODELDIR='./models/and'
MODELNAME='./models/and/and'

if __name__=='__main__':
    cmd=sys.argv[1]
    # data prepare
    if cmd=='create':
        input_data=tf.constant([[1,1],[1,0],[0,1],[0,0]])
        label_data=tf.constant([[1],[0],[0],[0]])
        dataset=tf.data.Dataset.from_tensor_slices((input_data,label_data))
        dataset=dataset.repeat() # inf data when the params is None
        dataset=dataset.shuffle(128)
        dataset=dataset.batch(4)
        it=dataset.make_one_shot_iterator()
        next_batch=it.get_next()
        counter=tf.Variable(0,trainable=False,name='global_step')
        # build graph with default
        inputs=tf.placeholder(tf.float64,shape=[None,2],name='inputs')
        lay1=tf.layers.Dense(units=8,activation=tf.nn.sigmoid,kernel_initializer=tf.random_uniform_initializer())(inputs)
        lay2=tf.layers.Dense(units=8,activation=tf.nn.sigmoid,kernel_initializer=tf.random_uniform_initializer())(lay1)
        outputs=tf.layers.Dense(units=1,activation=None,kernel_initializer=tf.random_uniform_initializer())(lay2)
        labels=tf.placeholder(tf.float64,shape=[None,1],name='labels')
        loss=tf.reduce_mean(tf.square(labels-outputs))
        optimize=tf.train.GradientDescentOptimizer(
            0.1).minimize(loss,global_step=counter)
        init=tf.global_variables_initializer()

        # tensorboard summary
        tf.summary.scalar('loss',loss)
        log=tf.summary.merge_all()

        writer=tf.summary.FileWriter(LOGDIR,tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(init)
            saver=tf.train.Saver()
            saver.save(sess,MODELNAME,global_step=counter)
    
    if cmd=='train':
        # load model
        checkpoint=tf.train.get_checkpoint_state(MODELDIR)
        if checkpoint is None or checkpoint.model_checkpoint_path is None:
            print("please run './use_pretrain_model.py create firstly'")
            exit()

        steps=sys.argv[2]
        saver=tf.train.import_meta_graph(checkpoint.model_checkpoint_path+'.meta')
        with tf.Session() as sess:
            saver.restore(sess,checkpoint.model_checkpoint_path)
            writer=tf.summary.FileWriter(LOGDIR,tf.get_default_graph())
            for i in range(int(steps)):
                x,y=sess.run(next_batch)
                _,summary=sess.run([optimize,log],feed_dict={inputs:x,labels:y})
                writer.add_summary(summary,global_step=counter.eval())
            saver.save(sess,MODELNAME,global_step=counter)

    if cmd=='test':
        checkpoint=tf.train.get_checkpoint_state(MODELDIR)
        if checkpoint is None or checkpoint.model_checkpoint_path is None:
            print("please run './use_pretrain_model.py create firstly'")
            exit()

        saver=tf.train.import_meta_graph(checkpoint.model_checkpoint_path+'.meta')
        with tf.Session() as sess:
            saver.restore(sess,checkpoint.model_checkpoint_path)
            x,y=sess.run(next_batch)
            res=sess.run(outputs,feed_dict={inputs:x,labels:y})
            print(res)
            print(y)
    
