#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
import numpy as np
import sys


# settings
LOGDIR = './log/learn/train'
# ./models must be created!!!
MODELDIR = './models/and'
MODELNAME = './models/and/and'

if __name__ == '__main__':
    cmd = sys.argv[1]

    if cmd == 'create':
        # data prepare
        input_data = tf.constant([[1, 1], [1, 0], [0, 1], [0, 0]])
        label_data = tf.constant([[1], [0], [0], [0]])
        dataset = tf.data.Dataset.from_tensor_slices((input_data, label_data))
        dataset = dataset.repeat()  # inf data when the params is None
        dataset = dataset.shuffle(128)
        dataset = dataset.batch(4)
        it = dataset.make_one_shot_iterator()
        next_batch = it.get_next(name='next_batch')

        counter = tf.Variable(0, trainable=False, name='global_step')
        # build graph with default
        inputs = tf.placeholder(tf.float64, shape=[None, 2], name='inputs')
        lay1 = tf.layers.Dense(units=8, activation=tf.nn.sigmoid,
                               kernel_initializer=tf.random_uniform_initializer())(inputs)
        lay2 = tf.layers.Dense(units=8, activation=tf.nn.sigmoid,
                               kernel_initializer=tf.random_uniform_initializer())(lay1)
        outputs = tf.layers.Dense(
            units=1, activation=None, kernel_initializer=tf.random_uniform_initializer())(lay2)
        labels = tf.placeholder(tf.int64, shape=[None, 1], name='labels')
        loss = tf.reduce_mean(tf.square(tf.cast(labels,tf.float64)-outputs))
        optimize = tf.train.GradientDescentOptimizer(
            0.1).minimize(loss, global_step=counter, name='optimize')

        # tensorboard summary
        tf.summary.scalar('loss', loss)
        log = tf.summary.merge_all(name='log')

        init = tf.global_variables_initializer()

        writer = tf.summary.FileWriter(LOGDIR, tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()
            saver.save(sess, MODELNAME, global_step=counter)

    if cmd == 'train':
        # load model
        checkpoint = tf.train.get_checkpoint_state(MODELDIR)
        if checkpoint is None or checkpoint.model_checkpoint_path is None:
            print("please run './use_pretrain_model.py create firstly'")
            exit()

        steps = sys.argv[2]
        saver = tf.train.import_meta_graph(
            checkpoint.model_checkpoint_path+'.meta')
        with tf.Session() as sess:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            writer = tf.summary.FileWriter(LOGDIR, tf.get_default_graph())

            # load nodes
            graph = tf.get_default_graph()

            # you need lookup tensors' name by next two command
            tensor_name_list = [
                tensor.name for tensor in graph.as_graph_def().node]
            print(tensor_name_list)
            next_x = graph.get_tensor_by_name('next_batch:0')
            next_y = graph.get_tensor_by_name('next_batch:1')
            inputs = graph.get_tensor_by_name('inputs:0')
            labels = graph.get_tensor_by_name('labels:0')
            counter = graph.get_tensor_by_name('global_step:0')
            optimize = graph.get_tensor_by_name('optimize:0')
            log = graph.get_tensor_by_name('log/log:0')

            for i in range(int(steps)):
                x, y = sess.run([next_x, next_y])
                _, summary = sess.run([optimize, log], feed_dict={
                                      inputs: x, labels: y})
                writer.add_summary(summary, global_step=counter.eval())
            saver.save(sess, MODELNAME, global_step=counter)

    if cmd == 'test':
        checkpoint = tf.train.get_checkpoint_state(MODELDIR)
        if checkpoint is None or checkpoint.model_checkpoint_path is None:
            print("please run './use_pretrain_model.py create firstly'")
            exit()

        saver = tf.train.import_meta_graph(
            checkpoint.model_checkpoint_path+'.meta')
        with tf.Session() as sess:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            graph = tf.get_default_graph()

            next_x = graph.get_tensor_by_name('next_batch:0')
            next_y = graph.get_tensor_by_name('next_batch:1')
            inputs = graph.get_tensor_by_name('inputs:0')
            labels = graph.get_tensor_by_name('labels:0')
            outputs = graph.get_tensor_by_name('dense_2/BiasAdd:0')

            x, y = sess.run([next_x, next_y])
            res = sess.run(outputs, feed_dict={inputs: x, labels: y})
            print(res)
            print(y)

    if cmd == "inhert_model":
        checkpoint = tf.train.get_checkpoint_state(MODELDIR)
        if checkpoint is None or checkpoint.model_checkpoint_path is None:
            print("please run './use_pretrain_model.py create firstly'")
            exit()

        saver = tf.train.import_meta_graph(
            checkpoint.model_checkpoint_path+'.meta')
        graph = tf.get_default_graph()
        next_x = graph.get_tensor_by_name('next_batch:0')
        next_y = graph.get_tensor_by_name('next_batch:1')
        inputs = graph.get_tensor_by_name('inputs:0')
        labels = graph.get_tensor_by_name('labels:0')
        outputs = graph.get_tensor_by_name('dense_2/BiasAdd:0')

        # define new layer
        with tf.variable_scope('new_layer'):
            new_output = tf.layers.Dense(2, activation=None)(outputs)
        predict=tf.nn.softmax(new_output)
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=new_output)


        
        # optimize the last layer
        update_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='new_layer')
        optz = tf.train.AdamOptimizer(0.01).minimize(
            loss, var_list=update_vars)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, checkpoint.model_checkpoint_path)

            # training new model
            for i in range(10000):
                x, y = sess.run([next_x, next_y])
                _, o, no = sess.run([optz, outputs, new_output], feed_dict={
                                    inputs: x, labels: y})

            # test
            x, y = sess.run([next_x, next_y])
            o, no = sess.run([outputs, predict],
                                feed_dict={inputs: x, labels: y})
            print(x)
            print(no)
            print(o)
