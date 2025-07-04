import tensorflow as tf
from fastkanTF import FastKAN
from tensorflow.keras import datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28)).astype('float32')
x_test = x_test.reshape((-1, 28*28)).astype('float32')

model = FastKAN([28*28, 64, 10])

model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

