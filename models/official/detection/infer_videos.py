# pylint: disable=line-too-long
r"""A stand-alone binary to run model inference and visualize results.
It currently only supports model of type `retinanet` and `mask_rcnn`. It only
supports running on CPU/GPU with batch size 1.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import flags
from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf

from official.detection.configs import factory as config_factory
from official.detection.dataloader import mode_keys, anchor
from official.detection.modeling import factory as model_factory
from official.detection.utils import input_utils
from hyperparameters import params_dict
from official.detection.utils.object_detection import visualization_utils

import cv2

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', 'retinanet', 'Support `retinanet`, `mask_rcnn` and `shapemask`.')
flags.DEFINE_integer('image_size', 640, 'The image size.')
flags.DEFINE_string('checkpoint_path', None, 'The path to the checkpoint file.')
flags.DEFINE_string('config_file', None, 'The config file template.')
flags.DEFINE_string(
    'params_override', '', 'The YAML file/string that specifies the parameters '
    'override in addition to the `config_file`.')
flags.DEFINE_string('video_file', None, 'Path of input video.')
flags.DEFINE_float(
    'min_score_threshold', 0.05,
    'The minimum score thresholds in order to draw boxes.')
flags.DEFINE_integer('target_class_id', 1, 'Target class id (default is 1, ie. COCO Person class).')


def get_frames(fname):
    cap = cv2.VideoCapture(fname)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break
    cap.release()


def _bb_to_csv(bb):
  return "%d %d %d %d %d %f" % tuple(bb)


def main(unused_argv):
  del unused_argv

  params = config_factory.config_generator(FLAGS.model)
  if FLAGS.config_file:
    params = params_dict.override_params_dict(
        params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.override({
      'architecture': {
          'use_bfloat16': False,  # The inference runs on CPU/GPU.
      },
  }, is_strict=True)
  params.validate()
  params.lock()

  model = model_factory.model_generator(params)

  with tf.Graph().as_default():
    image_raw = tf.placeholder(shape=(None, None, 3), dtype=tf.uint8)

    image = input_utils.normalize_image(image_raw)
    image_size = [FLAGS.image_size, FLAGS.image_size]
    image, image_info = input_utils.resize_and_crop_image(
        image,
        image_size,
        image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image.set_shape([image_size[0], image_size[1], 3])

    # batching.
    images = tf.reshape(image, [1, image_size[0], image_size[1], 3])
    images_info = tf.expand_dims(image_info, axis=0)

    # model inference
    outputs = model.build_outputs(
        images, {'image_info': images_info}, mode=mode_keys.PREDICT)
    outputs['detection_boxes'] = (
        outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]))

    predictions = outputs

    # Create a saver in order to load the pre-trained checkpoint.
    saver = tf.train.Saver()

    boxes = []
    with tf.Session() as sess:
      print(' - Loading the checkpoint...')
      saver.restore(sess, FLAGS.checkpoint_path)

      for i, frame in enumerate(get_frames(FLAGS.video_file)):
        print(' - Processing image %d...' % i)

        predictions_np = sess.run(
            predictions, feed_dict={image_raw: frame})

        num_detections = int(predictions_np['num_detections'][0])
        np_boxes = predictions_np['detection_boxes'][0, :num_detections]
        np_scores = predictions_np['detection_scores'][0, :num_detections]
        np_classes = predictions_np['detection_classes'][0, :num_detections]
        np_classes = np_classes.astype(np.int32)

        above_thresh = np_scores >= FLAGS.min_score_threshold
        category_filter = np_classes == FLAGS.target_class_id
        mask = above_thresh & category_filter
        np_boxes = np_boxes[mask]
        np_scores = np_scores[mask]
        np_classes = np_classes[mask]
        boxes.append(
            np.hstack((np_classes[:, np.newaxis], np_boxes.round().astype(np.int32), np_scores[:, np.newaxis])).tolist()
        )
        break

    out_csv_name = "%s.csv" % FLAGS.video_file
    with open(out_csv_name, "w") as fhan:
      for ind, bbs in enumerate(boxes):
        fhan.write("%d %s\n" % (ind, " ".join([_bb_to_csv(bb) for bb in bbs])))


if __name__ == '__main__':
  flags.mark_flag_as_required('config_file')
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('video_file')
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
