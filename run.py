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


from keras.preprocessing.image import img_to_array,load_img
import matplotlib.pyplot as plt
import cv2

base_model  = InceptionV3(weights = 'imagenet', include_top=False)

last = base_model.output
x = GlobalAveragePooling2D()(last)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(3,activation='softmax')(x)


model = Model(input=base_model.input,output=preds)
model.load_weights('/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/saved_model/fine_tuning.hdf5.hdf5')

def pred(img_path):    
    img = load_img(img_path,target_size = (299,299)) #Load the image and set the target size to the size of input of our model
    x = img_to_array(img) #Convert the image to array
    x = np.expand_dims(x,axis=0) #Convert the array to the form (1,x,y,z) 
    x = preprocess_input(x) # Use the preprocess input function o subtract the mean of all the images
    p = np.argmax(model.predict(x)) # Store the argmax of the predictions
    if p==0:     # P=0 for basal,P=1 for melanoma , P=2 for squamous
        print("basal")
    elif p==1:
        print("melanoma")
    elif p==2:
        print("squamous")

pred("/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/check image/basal.jpg")
z = plt.imread('/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/check image/basal.jpg') 
plt.imshow(z); 

pred("/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/check image/melanoma.jpg")
z = plt.imread('/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/check image/melanoma.jpg') 
plt.imshow(z);         #print the loaded image


pred("/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/check image/squamous.jpg")
z = plt.imread('/media/mango/DATA-2/Skin Cancer OE-DM Project-150905258/Skin-cancer-image-classification/check image/squamous.jpg') 
plt.imshow(z);         #print the loaded image




