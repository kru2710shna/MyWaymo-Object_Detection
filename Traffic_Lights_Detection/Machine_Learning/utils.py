import tensorflow as tf
import logging
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

def check_softmax(func):
    logits = tf.constant([[0.5, 1.0, 2.0, 0.3, 4.0]])
    tf_soft = tf.nn.softmax(logits)
    soft = func(logits)
    l1_norm = tf.norm(tf_soft - soft, ord=1)
    assert l1_norm < 1e-5, "Softmax calculation is wrong"
    print("Softmax implementation is correct!")


def check_ce(func):
    logits = tf.constant([[0.5, 1.0, 2.0, 0.3, 4.0]])
    scaled_logits = tf.nn.softmax(logits)
    one_hot = tf.constant([[0, 0, 0, 0, 1.0]])
    tf_ce = tf.nn.softmax_cross_entropy_with_logits(one_hot, logits)
    ce = func(scaled_logits, one_hot)
    l1_norm = tf.norm(tf_ce - ce, ord=1)
    assert l1_norm < 1e-5, "CE calculation is wrong"
    print("CE implementation is correct!")


def check_model(func):
    # only check the output size here
    X = tf.random.uniform([28, 28, 3])
    num_inputs = 28 * 28 * 3
    num_outputs = 10
    W = tf.Variable(
        tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01)
    )
    b = tf.Variable(tf.zeros(num_outputs))
    out = func(X, W, b)
    assert out.shape == (1, 10), "Model is wrong!"
    print("Model implementation is correct!")


def check_acc(func):
    y_hat = tf.constant([[0.8, 0.2, 0.5, 0.2, 5.0], [0.8, 0.2, 0.5, 0.2, 5.0]])
    y = tf.constant([4, 1])
    acc = func(y_hat, y)
    assert acc == tf.cast(tf.constant(0.5), dtype=acc.dtype), (
        "Accuracy calculation is wrong!"
    )
    print("Accuracy implementation is correct!")


def process(image, label):
    """small function to normalize input images"""
    image = tf.cast(image / 255.0, tf.float32)
    return image, label


def display_metrics(history):
    """plot loss and accuracy from keras history object"""
    f, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history.history["loss"], linewidth=3)
    ax[0].plot(history.history["val_loss"], linewidth=3)
    ax[0].set_title("Loss", fontsize=16)
    ax[0].set_ylabel("Loss", fontsize=16)
    ax[0].set_xlabel("Epoch", fontsize=16)
    ax[0].legend(["train loss", "val loss"], loc="upper right")
    ax[1].plot(history.history["accuracy"], linewidth=3)
    ax[1].plot(history.history["val_accuracy"], linewidth=3)
    ax[1].set_title("Accuracy", fontsize=16)
    ax[1].set_ylabel("Accuracy", fontsize=16)
    ax[1].set_xlabel("Epoch", fontsize=16)
    ax[1].legend(["train acc", "val acc"], loc="upper left")
    plt.show()


def get_datasets(imdir):
    """extract GTSRB dataset from directory"""
    train_dataset = image_dataset_from_directory(
        imdir,
        image_size=(32, 32),
        batch_size=256,
        validation_split=0.2,
        subset="training",
        seed=123,
        label_mode="int",
    )
    val_dataset = image_dataset_from_directory(
        imdir,
        image_size=(32, 32),
        batch_size=256,
        validation_split=0.2,
        subset="validation",
        seed=123,
        label_mode="int",
    )
    train_dataset = train_dataset.map(process)
    val_dataset = val_dataset.map(process)
    return train_dataset, val_dataset


def get_module_logger(mod_name):
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger



def check_output(output):
    if output.shape == (1, 75, 75, 16):
        print('Success!')
    else:
        print('Failure')
        
def processCNN(image,label):
    """ small function to normalize input images """
    image = tf.cast(image/255. ,tf.float32)
    return image,label



def display_metricsCNN(history):
    """ plot loss and accuracy from keras history object """
    f, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history.history['loss'], linewidth=3)
    ax[0].plot(history.history['val_loss'], linewidth=3)
    ax[0].set_title('Loss', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].set_xlabel('Epoch', fontsize=16)
    ax[0].legend(['train loss', 'val loss'], loc='upper right')
    ax[1].plot(history.history['accuracy'], linewidth=3)
    ax[1].plot(history.history['val_accuracy'], linewidth=3)
    ax[1].set_title('Accuracy', fontsize=16)
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].set_xlabel('Epoch', fontsize=16)
    ax[1].legend(['train acc', 'val acc'], loc='upper left')
    plt.show()
    
    
def get_datasetsCNN(imdir):
    """ extract GTSRB dataset from directory """
    train_dataset = image_dataset_from_directory(imdir, 
                                       image_size=(32, 32),
                                       batch_size=32,
                                       validation_split=0.2,
                                       subset='training',
                                       seed=123,
                                       label_mode='int')

    val_dataset = image_dataset_from_directory(imdir, 
                                        image_size=(32, 32),
                                        batch_size=32,
                                        validation_split=0.2,
                                        subset='validation',
                                        seed=123,
                                        label_mode='int')
    train_dataset = train_dataset.map(processCNN)
    val_dataset = val_dataset.map(processCNN)
    return train_dataset, val_dataset


def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin + 1) * max(0, ymax - ymin + 1)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union


def check_results(output):
    # ✅ Corrected relative path from Machine_Learning folder
    truth = np.load('../Data/nms.npy', allow_pickle=True)
    assert np.array_equal(truth, np.array(output, dtype="object")), 'The NMS implementation is wrong'
    print('The NMS implementation is correct!')