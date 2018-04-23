from keras.applications.inception_v3 import InceptionV3,preprocess_input,decode_predictions
from keras.preprocessing import image
import numpy as np
from keras.layers import Dense, GlobalAveragePooling2D,Dropout,Input
# from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.models import Sequential,Model
from keras import backend as K
from IPython.display import display
from keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
import cv2


def plot_training(history):
    acc = history.history['acc'] 
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs  = range(len(acc))
    
    plt.plot(epochs,acc,'b')
    plt.plot(epochs,val_acc,'r')
    plt.title("Training and validation accuracy")
    
    plt.figure()
    plt.plot(epochs,loss,'b')
    plt.plot(epochs,val_loss,'r')
    plt.title("Training and validation loss")
    
    plt.show()

base_model  = InceptionV3(weights = 'imagenet', include_top=False)
print('loaded model')

data_gen_args = dict(rescale=1.0/255.0, #Define the dictionary for Image data Generator
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip = True)

train_datagen = image.ImageDataGenerator(**data_gen_args)
test_datagen = image.ImageDataGenerator(**data_gen_args)



train_generator = train_datagen.flow_from_directory("/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/train",
                                                    target_size=(299,299),batch_size=32,
                                                    class_mode='categorical',
                                                    classes=['basal','melanoma','squamous'])

valid_generator = test_datagen.flow_from_directory("/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/test",
                                                     target_size=(299,299),batch_size=32,
                                                    class_mode='categorical',
                                                    classes=['basal','melanoma','squamous'])

from keras.layers import Conv2D,MaxPooling2D,Flatten

benchmark = Sequential()
benchmark.add(Conv2D(filters = 16, kernel_size = 2, padding = 'same', activation = 'relu', input_shape = (299,299,3)))
benchmark.add(MaxPooling2D(pool_size=2,padding='same'))
benchmark.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
benchmark.add(MaxPooling2D(pool_size=2,padding='same'))
benchmark.add(Conv2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu'))
benchmark.add(MaxPooling2D(pool_size=2,padding='same'))
benchmark.add(Dropout(0.3))
benchmark.add(Flatten())
benchmark.add(Dense(512, activation='relu'))
benchmark.add(Dropout(0.5))
benchmark.add(Dense(3, activation='softmax'))

benchmark.summary()


benchmark.compile(loss = 'categorical_crossentropy',optimizer='rmsprop', metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint,EarlyStopping

# Save the model with best weights
checkpointer = ModelCheckpoint('/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/saved_model/benchmark.hdf5', verbose=1,save_best_only=True)
# Stop the training if the model shows no improvement 
stopper = EarlyStopping(monitor='val_loss',min_delta=0.1,patience=0,verbose=1,mode='auto')

history = benchmark.fit_generator(train_generator, steps_per_epoch = 13,validation_data=valid_generator,validation_steps=3, epochs=5,verbose=1,callbacks=[checkpointer])



# Define the output layers for Inceptionv3
last = base_model.output
x = GlobalAveragePooling2D()(last)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(3,activation='softmax')(x)

model = Model(input=base_model.input,output=preds)
model.summary()

#Load the weights for the common layers from the benchmark model
base_model.load_weights(filepath='/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/saved_model/benchmark.hdf5',by_name=True)

#Freeze the original layers of Inception3
for layer in base_model.layers:
    layer.trainable = False

#Compile the model
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint,EarlyStopping

# Save the model with best weights
checkpointer = ModelCheckpoint('/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/saved_model/transfer_learning.hdf5', verbose=1,save_best_only=True)
# Stop the traning if the model shows no improvement
stopper = EarlyStopping(monitor='val_loss',min_delta=0.1,patience=1,verbose=1,mode='auto')

# Train the model
history_transfer = model.fit_generator(train_generator, steps_per_epoch = 13,validation_data=valid_generator,validation_steps=4, epochs=5,verbose=1,callbacks=[checkpointer])

display(history_transfer.history)

plot_training(history_transfer)

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)


# Unfreeze the last three inception modules
for layer in model.layers[:229]:
    layer.trainable = False
for layer in model.layers[229:]:
    layer.trainable = True

from keras.optimizers import SGD

# Use an optimizer with slow learning rate
model.compile(optimizer=SGD(lr=0.0001,momentum=0.9),loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Save the model with best validation loss
checkpointer = ModelCheckpoint('/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/saved_model/fine_tuning.hdf5.hdf5', verbose=1,save_best_only=True,monitor='val_loss')

# Stop the traning if the validation loss doesn't improve
stopper = EarlyStopping(monitor='val_loss,val_acc',min_delta=0.1,patience=2,verbose=1,mode='auto')

# Train the model
history = model.fit_generator(train_generator, steps_per_epoch = 13,validation_data=valid_generator,validation_steps=3, epochs=5,verbose=1,callbacks=[checkpointer])






