#Build Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import keras

batch_size = 64
num_classes = 8
epochs = 15

def build_model(optimizer):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=keras.regularizers.l1(0.1)))
    model.add(keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    return model

# Assuming input_shape is defined earlier in your code
# input_shape = ...

optimizers = ['Adagrad']
for i in optimizers:
    model = build_model(i)
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# The rest of your code for saving and plotting remains the same.


import pandas as pd
hist_df = pd.DataFrame(hist.history)
# or save to csv:
hist_csv_file = '/content/drive/MyDrive/tomato_pest/adagrad_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
#plotting
from matplotlib import pyplot as plt
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
# plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'val'], loc='upper left')
plt.show()