#!/bin/bash

nvidia-docker run -it --rm -e PYTHONWARNINGS="ignore" nvcr.io/nvidia/tensorflow:24.10-tf2-py3 python -c "
import tensorflow as tf
import time
print('TensorFlow Version:', tf.__version__)
print('CUDA Available:', tf.config.list_physical_devices('GPU') != [])

if tf.config.list_physical_devices('GPU'):
    gpu_name = tf.config.list_physical_devices('GPU')[0].name
    print('CUDA Device Name:', gpu_name)
    print('CUDA Device Count:', len(tf.config.list_physical_devices('GPU')))

# CPU test
cpu_tensor = tf.random.uniform((10000, 10000))
start_time = time.time()
cpu_sum = tf.reduce_sum(cpu_tensor)
end_time = time.time()
print('CPU Sum:', cpu_sum.numpy())
print('CPU Time taken: {:.6f} seconds'.format(end_time - start_time))

# GPU test
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        gpu_tensor = tf.random.uniform((10000, 10000))
        start_time = time.time()
        gpu_sum = tf.reduce_sum(gpu_tensor)
        end_time = time.time()
        print('GPU Sum:', gpu_sum.numpy())
        print('GPU Time taken: {:.6f} seconds'.format(end_time - start_time))
else:
    print('GPU is not available.')
"
