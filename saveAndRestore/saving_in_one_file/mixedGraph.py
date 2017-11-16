import tensorflow as tf

vgg_saver = tf.train.import_meta_graph(dir + '/result/vgg-16.meta')

vgg_graph = tf.get_default_graph()

self.x_plh = vgg_graph.get_tensor_by_name('input:0')

output_conv = vgg_graph.get_tensor_by_name('conv1_2:0')

output_conv_sg = tf.stop_gradient(output_conv)


output_conv_shape = output_conv_sg.get_shape().as_list()

W1 = tf.get_variable('W1', shape=[1, 1, output_conv_shape[3], 32],
  initializer=tf.random_normal_initializer(stddev=1e-1))
b1 = tf.get_variable('b1', shape=[32], initializer=tf.constant_initializer(0.1))
z1 = tf.nn.conv2d(output_conv_sg, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
a = tf.nn.relu(z1)


