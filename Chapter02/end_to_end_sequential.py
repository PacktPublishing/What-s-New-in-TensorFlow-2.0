import tensorflow as tf
import tensorflow.keras as keras

# Load Data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Convert data from numpy arrays to tensors
x_train = tf.convert_to_tensor(
    x_train,
    dtype='float'
)

y_train = tf.one_hot(
    y_train,
    depth=10
)


x_test = tf.convert_to_tensor(
    x_test,
    dtype='float'
)
y_test = tf.one_hot(
    y_test,
    depth=10
)


# Build neural network for classification
model = keras.Sequential()


model.add(
    keras.layers.Flatten()
)

model.add(
    keras.layers.Dense(100,
                       activation='relu',
                       )
)
model.add(
    keras.layers.Dense(100,
                       activation='relu'
                       )
)
model.add(
    keras.layers.Dense(10,
                       activation='softmax')
)

# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

# Train model
model.fit(x_train,
          y_train,
          batch_size=32,
		  steps_per_epoch=5,
          epochs=10)


# Run evaluation
score = model.evaluate(x_test,
                       y_test
                       )
print('Test loss:', score[0])
print('Test accuracy:', score[1])
