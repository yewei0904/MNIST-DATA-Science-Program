# MNIST-DATA-Science-Program

## Introduction
In this project, I mainly learn how to save and read data. After training the network, it wants to save the trained model and read in the program to save the trained model.

## DEMO
![img](https://github.com/yewei0904/MNIST-DATA-Science-Program/blob/master/gifhome_1920x1080_10s.gif)

## Background
To save and restore, you need to instantiate a tf.train.saver.
  saver = tf.train.Saver()
Then, in the training loop, you periodically call the saver. Save () method to write to the folder the checkpoint file that contains all the trainable variables in the current model.
  saver.save(sess, FLAGS.train_dir, global_step=step)
