import tensorflow as tf
import os

dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'result')

v1 =  tf.Variable(1., name="v1")
v2 =  tf.Variable(2., name="v2")

a = tf.add(v1,v2,name="a")

b = tf.multiply(a, 10, name='b')

print(a.graph == tf.get_default_graph())

all_saver = tf.train.Saver()

v2_saver = tf.train.Saver({'v2':v2})

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  all_saver.save(sess, dir+'/data_all')
  #v2_saver.save(sess, dir+'/data_v2')
  print([n.name for n in tf.get_default_graph().as_graph_def().node])
