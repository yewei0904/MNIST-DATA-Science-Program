# MNIST-DATA-Science-Program

## Introduction
In this project, I mainly learn how to save and read data. After training the network, it wants to save the trained model and read in the program to save the trained model.

## DEMO
![img](https://github.com/yewei0904/MNIST-DATA-Science-Program/blob/master/gifhome_1920x1080_10s.gif)

## Background
To save and restore, you need to instantiate a tf.train.saver.
```python
saver = tf.train.Saver()
```
Then, in the training loop, you periodically call the saver. Save () method to write to the folder the checkpoint file that contains all the trainable variables in the current model.
```python
saver.save(sess, FLAGS.train_dir, global_step=step)
```
After that, you can use the saver. Restore () method to override the model's parameters and continue training or testing the data.
```python
saver.restore(sess, FLAGS.train_dir)
```
After saver.save (), you can see the four new files in your folder.
