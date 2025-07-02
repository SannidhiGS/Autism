import os
import cv2
import numpy as np

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array, array_to_img
from keras.preprocessing import image

import pickle

image = cv2.imread("1.jpg")

#color conversion
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('static/gray.jpg', gray_image)
#apply the Canny edge detection
edges = cv2.Canny(image, 250, 254)
cv2.imwrite('static/edges.jpg', edges)
#apply thresholding to segment the image
retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
cv2.imwrite('static/threshold.jpg', threshold2)





model=load_model('autism_mri.h5')



# Load the class names
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

# Function to preprocess the input image
def preprocess_input_image(path):
    img = load_img(path, target_size=(150,150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Function to make predictions on a single image
def predict_single_image(path):
    input_image = preprocess_input_image(path)
    prediction = model.predict(input_image)
    print(prediction)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index]

    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
        
    return predicted_class, confidence 

predicted_class, confidence = predict_single_image(image)
#predicted_class, confidence = predict_single_image(path, model, class_names)
print(predicted_class, confidence)
     