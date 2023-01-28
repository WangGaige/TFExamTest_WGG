#TensorFlow 2 quickstart for beginners

# Set up TensorFlow
import tensorflow as tf
import tensorflow.keras as keras
# Load a dataset
mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255, x_test/255

# Build a machine learning model
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])
# Train and evaluate your model
model.fit(x_train,y_train,epochs=5)

a=model.evaluate(x_test,y_test,verbose=2)

print(a)