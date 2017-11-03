import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

slim = tf.contrib.slim

FLAGS = None

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob, is_training). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """

  keep_prob = tf.placeholder(tf.float32)
  is_training = tf.placeholder(tf.bool)

  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        biases_initializer=tf.constant_initializer(0.1)):
    h_conv1 = slim.conv2d(x_image, 32, 5, scope='conv1')
    h_pool1 = slim.max_pool2d(h_conv1, 2, padding='SAME', scope='pool1')
    h_conv2 = slim.conv2d(h_pool1, 64, 5, scope='conv2')
    h_pool2 = slim.max_pool2d(h_conv2, 2, padding='SAME', scope='pool2')
    net = slim.flatten(h_pool2, scope='flatten1')
    h_fc1 = slim.fully_connected(net, 1024, scope='fc1')
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    h_fc1_drop = slim.dropout(h_fc1, keep_prob=keep_prob, is_training=is_training, scope='dropout1')
    y_conv = slim.fully_connected(h_fc1_drop, 10, scope='fc2',activation_fn=None)

  return y_conv, keep_prob, is_training


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        ps_device="/job:ps/task:0/cpu:0",
        cluster=cluster)):
      # Build model...
      # Import data
      mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
      global_step = tf.contrib.framework.get_or_create_global_step()
      # Create the model
      x = tf.placeholder(tf.float32, [None, 784])
      # Define loss and optimizer
      y_ = tf.placeholder(tf.float32, [None, 10])
      # Build the graph for the deep net
      y_conv, keep_prob, is_training = deepnn(x)
      with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
      cross_entropy = tf.reduce_mean(cross_entropy)
      with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
      with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdagradOptimizer(0.01).minimize(cross_entropy
          , global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=20000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="./train_logs",
                                           hooks=hooks) as mon_sess:
      i=0
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        i+=1
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
          train_accuracy = mon_sess.run( accuracy, feed_dict={
              x: batch[0], y_: batch[1], keep_prob: 1.0, is_training: False})
          print('step %d, training accuracy %g' % (i, train_accuracy))
        mon_sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, is_training: True})

        if i == 19998:
          print('test accuracy %g' % mon_sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, is_training: False}))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  parser.add_argument('--data_dir', type=str,
                      default='./mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



'''
If your parameters are not sharded, you could do it with a simplified version of replica_device_setter like below:

def assign_to_device(worker=0, gpu=0, ps_device="/job:ps/task:0/cpu:0"):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op == "Variable":
            return ps_device
        else:
            return "/job:worker/task:%d/gpu:%d" % (worker, gpu)
    return _assign

with tf.device(assign_to_device(1, 2)):
  # this op goes on worker 1 gpu 2
  my_op = tf.ones(())
'''

'''
setting CUDA_VISIBLE_DEVICES appropriately could make the script see or not see gpu devices.

"set CUDA_VISIBLE_DEVICES=" could set empty to variable CUDA_VISIBLE_DEVICES.


It looks like the problem is in your worker_device="/gpu:%d" % (FLAGS.task_id%4) argument to tf.train.replica_device_setter().
There are two parts to the problem:

The device string doesn't specify a task ID (i.e. "/task:%d" % (FLAGS.task_id)). Unless you have specified device_filters in
your session configuration, this will result in all ops being placed in task 0, which runs in server A.
Each process on a particular server is creating devices "/gpu:0", ... ,"/gpu:3", because by default a server (or a single-process
tf.Session) will create one TensorFlow device per physical device on the system. This will lead to inefficient memory allocation
between the processes. You should use the CUDA_VISIBLE_DEVICES environment variable to limit each server to being able to see only
a single device, which will be available as "/gpu:0" in that process.
After setting CUDA_VISIBLE_DEVICES appropriately, you can use worker_device="/job:worker/task:%d/gpu:0" % (FLAGS.task_id) as the
argument to tf.train.replica_device_setter(), and the utilization should be balanced across the GPUs (assuming that you build the
same graph in each of your worker processes, and use something like the tf.train.Supervisor to manage the distributed execution).
'''
