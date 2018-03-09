#Name: Marathe Maitreyee Sanjiv
#Roll No.: 14EE129



import numpy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import regularizers

# random seed
seed = 50
numpy.random.seed(seed)

# load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten each 28*28 image to a single 784 vector
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

# Normalization of inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# One hot encoding of outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Define the baseline model
def baseline_model():
	sgd = SGD(momentum=0.9)
	model = Sequential()
	model.add(Dense(784, input_dim=num_pixels, kernel_initializer='normal', kernel_regularizer=regularizers.l1(0.01), activation='relu')) #kernel regularizer => weight #regularizer
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return model

# Build model
model = baseline_model()
# Fit the model
History = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=100, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy: %.2f%%" % (scores[1]*100))

#Plotting graphs

epochs = range(10)
plt.figure(1) 
plt.plot(epochs,History.history['acc'],label='acc')
plt.xlabel('Epochs')
plt.ylabel('Percentage')
plt.title('Training Accuracy')
 
plt.figure(2) 
plt.plot(epochs,History.history['loss'],label='loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')


plt.figure(3)
plt.plot(epochs,History.history['val_acc'])
plt.xlabel('Epochs')
plt.ylabel('Percentage')
plt.title('Validation Accuracy')
 

plt.figure(4)
plt.plot(epochs,History.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss')


plt.show()