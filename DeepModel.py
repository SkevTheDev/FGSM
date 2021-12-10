from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

class DeepModel(object):
    def create_model(self, img_rows, img_cols, channels, num_classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu', input_shape=(img_rows, img_cols, channels)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(32))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
        return model

    def train_model(self, model, x_train, y_train, x_test, y_test, epochs, batch_size):
        model.fit(x_train, y_train,
          batch_size=32,
          epochs=epochs,
          validation_data=(x_test, y_test))
        return model

