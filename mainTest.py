import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.h5')

image_path = r'C:\Users\rishu\Downloads\BRAIN TUMOR IMAGE CLASSIFICATION\pred\pred0.jpg'
image = cv2.imread(image_path)

if image is not None:
    img = Image.fromarray(image)
    img = img.resize((64, 64))
    img = np.array(img)
    
    input_img = np.expand_dims(img, axis=0)
    
    # Use the predict method to get prediction probabilities
    predictions = model.predict(input_img)
    
    # Get the index of the class with the highest probability
    predicted_class = np.argmax(predictions, axis=-1)
    
    print("Predicted class:", predicted_class)
else:
    print("Failed to load the image.")






