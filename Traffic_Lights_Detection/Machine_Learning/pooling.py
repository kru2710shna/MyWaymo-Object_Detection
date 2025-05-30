import argparse

import numpy as np

from utils import check_output


def get_paddings(array, pool_size, pool_stride):
    """ 
    get padding sizes 
    args:
    - array [array]: input np array NxwxHxC
    - pool_size [int]: window size
    - pool_stride [int]: stride
    returns:
    - paddings [list[list]]: paddings in np.pad format
    """
    _, w, h, _ = array.shape
    wpad = (w // pool_stride) * pool_stride + pool_size - w
    hpad = (h // pool_stride) * pool_stride + pool_size - h
    return [[0, 0], [0, wpad], [0, hpad], [0, 0]]


def get_output_size(shape, pool_size, pool_stride):
    """ 
    given input shape, pooling window and stride, output shape 
    args:
    - shape [list]: input shape
    - pool_size [int]: window size
    - pool_stride [int]: stride
    returns
    - output_shape [list]: output array shape
    """
    w = shape[1]
    h = shape[2]
    new_w = (w - pool_size) / pool_stride + 1
    new_h = (h - pool_size) / pool_stride + 1
    return [shape[0], int(new_w), int(new_h), shape[3]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and process tf files')
    parser.add_argument('-f', '--pool_size', required=True, type=int, default=3,
                        help='pool filter size')
    parser.add_argument('-s', '--stride', required=True, type=int, default=3,
                        help='stride size')
    args = parser.parse_args()   

    input_array = np.random.rand(1, 148, 148, 16)
    pool_size = args.pool_size
    pool_stride = args.stride

    # padd the input layer
    paddings = get_paddings(input_array, pool_size, pool_stride)
    padded = np.pad(input_array, paddings, mode='constant', constant_values=0)

    # get output size
    output_size = get_output_size(padded.shape, pool_size, pool_stride)
    output = np.zeros(output_size)

    idx = 0
    for i in range(0, input_array.shape[1], pool_stride):
        jdx = 0
        for j in range(0, input_array.shape[2], pool_stride):
            local = padded[:, i: i + pool_stride, j: j + pool_stride, :]
            local_max = np.max(local, axis=(1, 2))
            output[:, idx, jdx, :] = local_max
            jdx += 1
        idx += 1
    check_output(output)