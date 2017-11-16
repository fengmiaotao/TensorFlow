import tensorflow as tf

saver = tf.train.import_meta_graph("result/data_all.meta")

graph = tf.get_default_graph()

#print(graph.as_graph_def())
print([n.name for n in graph.as_graph_def().node])
print([n.name for n in tf.trainable_variables()])

v1 = graph.get_tensor_by_name('v1:0')
v2 = graph.get_tensor_by_name('v2:0')

#print(v1.shape, v2.shape)

with tf.Session() as sess:
  saver.restore(sess, "result/data_all")

  print('v1 is {}'.format(sess.run(v1)))


