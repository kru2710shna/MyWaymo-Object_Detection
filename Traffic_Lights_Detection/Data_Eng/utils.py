import json
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np


def check_results(ious):
    # check the results
    solution = np.load('data/exercise1_check.npy')
    assert (ious == solution).sum() == 40, 'The iou calculation is wrong!'
    print('Congrats, the iou calculation is correct!')


def get_data():
    """ simple wrapper function to get data """
    with open('data/ground_truth.json') as f:
        ground_truth = json.load(f)
    
    with open('data/predictions.json') as f:
        predictions = json.load(f)

    return ground_truth, predictions


def parse_frame(frame, camera_name='FRONT'):
    """ take a frame, output bbox + image"""
    # get image
    images = frame.images
    for im in images:
        if open_dataset.CameraName.Name.Name(im.name) != camera_name:
            continue
        encoded_jpeg = im.image
    
    # get bboxes
    labels = frame.camera_labels
    for lab in labels:
        if open_dataset.CameraName.Name.Name(lab.name) != camera_name:
            continue
        annotations = lab.labels
    return encoded_jpeg, annotations


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
