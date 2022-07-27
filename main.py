import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading dataset
mnist_dataset = tf.keras.datasets.mnist
(train_X, train_y), (test_X, test_y) = mnist_dataset.load_data()

# Normalizing pixels (not digits) in dataset to values in range [0, 1]
train_X = tf.keras.utils.normalize(train_X, axis=1)
test_X = tf.keras.utils.normalize(test_X, axis=1)

def generate_model():
    # Creating the model
    ann_model = tf.keras.models.Sequential()
    ann_model.add(tf.keras.layers.Flatten(input_shape=(28, 28))) # Creates an array (not a matrix) of size 784
    ann_model.add(tf.keras.layers.Dense(128, activation='relu')) # Hidden layer
    ann_model.add(tf.keras.layers.Dense(128, activation='relu')) # Hidden layer
    ann_model.add(tf.keras.layers.Dense(10, activation='softmax')) # This represents our output layer and softmax means all 10 values add up to 1. Gives us probability for each digit to be the right answer

    # Compiling the model
    ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Fitting the model, i.e. training the model
    ann_model.fit(train_X, train_y, epochs=3) # Epochs specify how many times the ANN will see the same data.

    # Saving the model
    ann_model.save('HWDRANN.model')

def use_loaded_model():
    # Load saved model
    ann_model = tf.keras.models.load_model('HWDRANN.model')

    # Evaluate model: Want low loss and high accuracy
    loss, accuracy = ann_model.evaluate(test_X, test_y)
    print('Loss: {}'.format(loss))
    print('Accuracy: {}'.format(accuracy))

    # Loading images of handwritten digits
    image_number = 0
    while os.path.isfile('images/digit{}.png'.format(image_number)):
        try:
            image = cv2.imread('images/digit{}.png'.format(image_number))[:,:,0] # Only care about the first channel, the shape of the number
            image = np.invert(np.array([image]))
            prediction = ann_model.predict(image)
            print('This digit is probably a {}'.format(np.argmax(prediction)))
            plt.imshow(image[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print('The resolution of the image is probably incorrect')
        finally:
            image_number += 1

if __name__ == '__main__':
    if os.path.isdir('HWDRANN.model'):
        print('||||||||||||||| USE SAVED MODEL |||||||||||||||')
        use_loaded_model()
    else:
        print('||||||||||||||| GENERATE NEW MODEL |||||||||||||||')
        generate_model()