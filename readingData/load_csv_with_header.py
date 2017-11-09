from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

#Data sets
IRIS_TRAINING =  "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_training.csv"

def main():

  if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING,"wb") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST,'wb') as f:
      f.write(raw)

  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING, target_dtype=np.int, features_dtype=np.float32)

  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST, target_dtype=np.int, features_dtype=np.float32)

  feature_columns = [tf.contrib.layers.real_valued_column("",dimension=4)]

  classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_columns,
      hidden_units=[10,20,10], n_classes=3, model_dir="/tmp/iris_model")

  classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

  accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)['accuracy']

  print('Accuracy : {0:f}'.format(accuracy_score))


if __name__ == '__main__':
  main()

