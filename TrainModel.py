import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 30
num_classes = 12

#leaky relu
tmodel = Sequential()

tmodel.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(50,50,2),padding='same'))
tmodel.add(MaxPooling2D((2, 2),padding='same'))
tmodel.add(BatchNormalization())
tmodel.add(LeakyReLU(alpha=0.01))
tmodel.add(Dropout(0.2))

tmodel.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
tmodel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
tmodel.add(BatchNormalization())
tmodel.add(LeakyReLU(alpha=0.01))
tmodel.add(Dropout(0.3))

tmodel.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
tmodel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
tmodel.add(BatchNormalization())
tmodel.add(LeakyReLU(alpha=0.01))
tmodel.add(Dropout(0.4))

tmodel.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
tmodel.add(LeakyReLU(alpha=0.01))
tmodel.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
tmodel.add(BatchNormalization())
tmodel.add(Dropout(0.5))

tmodel.add(Flatten()) 

tmodel.add(Dense(256, activation='relu'))
tmodel.add(BatchNormalization())
tmodel.add(LeakyReLU(alpha=0.01))
tmodel.add(Dropout(0.5))     

tmodel.add(Dense(num_classes, activation='softmax'))

tmodel.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
ttrain_dropout = tmodel.fit(trainX, trainY, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(validateX, validateY))
