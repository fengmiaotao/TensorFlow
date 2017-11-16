import tensorflow as tf

import os, argparse

dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, output_node_names):
  """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
  """

  if not tf.gfile.Exists(model_dir):
    raise AssertionError(
      "Export directory doesn't exists. Please specify an export "
      "directory: %s" % model_dir)

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  checkpoint = tf.train.get_checkpoint_state(model_dir)
  input_checkpoint = checkpoint.model_checkpoint_path
  print(input_checkpoint)

  absolute_model_dir = '/'.join(input_checkpoint.split('/')[:-1])
  output_graph = os.path.join(absolute_model_dir,'freeze_model.pb')

  clear_devices =True

  with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    saver.restore(sess, input_checkpoint)

    print([n.name for n in tf.get_default_graph().as_graph_def().node])

    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess,
      tf.get_default_graph().as_graph_def(),
      output_node_names.split(',')

      )

    with tf.gfile.GFile(output_graph, 'wb') as f:
      f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

  return output_graph_def

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
  parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
  args = parser.parse_args()
  freeze_graph(args.model_dir, args.output_node_names)










