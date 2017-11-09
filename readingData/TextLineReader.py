
import tensorflow as tf
import os


def read_data(file_queue):
  reader = tf.TextLineReader(skip_header_lines=1)
  key, value = reader.read(file_queue)

  defaults = [[0.],[0.],[0.],[0.],[""]]

  SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species = tf.decode_csv(
      value, defaults)

  preprocess_op = tf.case({
    tf.equal(Species, tf.constant('Iris-setosa')): lambda: tf.constant(0),
    tf.equal(Species, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
    tf.equal(Species, tf.constant('Iris-virginica')): lambda: tf.constant(2),
    }, lambda: tf.constant(-1), exclusive=True)

  return tf.stack([SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]), preprocess_op

def create_pipeline(filename, batch_size, num_epochs=None):
  file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
  example, label = read_data(file_queue)

  min_after_dequeue = 10
  capacity = min_after_dequeue +batch_size
  example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size
    )

  return example_batch, label_batch,

x_train_batch, y_train_batch = create_pipeline('./iris_training.csv', 5, num_epochs=100)
x_test, y_test = create_pipeline('./iris_test.csv', 6)
#print(x_train_batch,y_train_batch)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print(sess.run([x_test]))

  coord.request_stop()
  coord.join(threads)
