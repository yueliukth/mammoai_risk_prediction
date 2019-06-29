# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '.',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '.', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 8,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'risk', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'val', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '.', 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_resnet_v2', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', 299, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
#    [image, label, basename] = provider.get(['image', 'label', 'basename'])
    [image, label, basename, age, exposure, current, thickness, compression] = provider.get(['image', 'label','basename', 'age', 'exposure', 'current', 'thickness', 'compression'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    def add_parameter(im,param):
        param = param*tf.ones(tf.shape(im[:,:,0]),dtype=im.dtype)
        param = tf.expand_dims(param,axis=2)
        output = tf.concat([im,param],axis=2)
        return output

    image = add_parameter(image,age)
    image = add_parameter(image,exposure)
    image = add_parameter(image,current)
    image = add_parameter(image,thickness)
    image = add_parameter(image,compression)

    images, labels, basenames = tf.train.batch(
        [image, label, basename],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    probabilities = tf.nn.softmax(logits)
    #print ('shape of labels before squeeze: ', tf.shape(labels))
    labels = tf.squeeze(labels)
    #print ('shape of labels after squeeze: ', tf.shape(labels))


    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = int(FLAGS.max_num_batches)
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = int(math.ceil(dataset.num_samples / float(FLAGS.batch_size)))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    sess_config=tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=sess_config) as sess:
                    
        tf.train.start_queue_runners()

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        restorer = tf.train.Saver(var_list=vars)
        restorer.restore(sess, checkpoint_path)

        output_file_path = os.path.join(FLAGS.eval_dir, 'preds_val.txt')
        of = open(output_file_path, 'w')

        accuracy = 0; tp = 0; fp = 0; tn = 0; fn = 0; y_true = []; y_scores = []; 
        start = time.time()

        for step_val in range(num_batches):
            print(step_val)
            preds, labs, probs, names = sess.run([predictions, labels, probabilities, basenames])
	    probs = probs[:,1]
	    
            for i in range(len(preds)):
            	of.write('%s %.12f %d\n' % (names[i], probs[i], labs[i]))
	    
	    y_true = np.append(y_true, labs);
	    y_scores = np.append(y_scores, probs);
	    pos = np.equal(labs, 1);
	    neg = np.invert(pos);
	    pred_pos = np.equal(preds, 1);
	    pred_neg = np.invert(pred_pos);
	    tp += np.sum(np.multiply(pos, pred_pos));
	    tn += np.sum(np.multiply(neg, pred_neg));
	    fp += np.sum(np.multiply(neg, pred_pos));
	    fn += np.sum(np.multiply(pos, pred_neg));
	auc = roc_auc_score(y_true, y_scores);
        
        accuracy = (tp+tn)/(tp+tn+fp+fn);
        logging.info('Accuracy: %.2f%%, TP: %d, FP: %d, TN: %d, FN: %d, AUC: %.2f%%', accuracy * 100, tp, fp, tn, fn, auc*100)
        end = time.time()
	print(auc)
        logging.info('Took %f seconds for evaluation', end-start);

	
        of.close()


if __name__ == '__main__':
  tf.app.run()
