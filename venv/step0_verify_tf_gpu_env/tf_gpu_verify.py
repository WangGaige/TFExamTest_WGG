import tensorflow as tf
print("TF version:")
print(tf.__version__)

gpus=tf.config.experimental.list_physical_devices(device_type='GPU')
cpus=tf.config.experimental.list_physical_devices(device_type='CPU')
print(gpus)
print(cpus)