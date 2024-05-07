# Define global variables
GLOB = 'Aero-engine_defect-detect_new/images'
SIZE = 512
STOP = 500
THUMBNAIL_SIZE = (100, 100)

import pickle
from PIL import Image
from img2vec_pytorch import Img2Vec
import numpy as np

model = Img2Vec(cuda=False, model='resnet-18')

# Load the saved model from file
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Preprocess new data
def preprocess_data(image_path):
    # Load image
    with Image.open(image_path) as img:
        img = img.convert('RGB')  # Ensure RGB mode
        img = img.resize((512, 512))  # Resize to required dimensions

    # Convert image to embedding vector
    img2vec = Img2Vec(cuda=False, model='resnet-18')
    img_embedding = img2vec.get_vec(img, tensor=True).numpy().reshape(512,)

    return img_embedding


# Example of how to use the preprocessing function
# Replace 'image_path' with the path to your new image
new_image_path = 'new_input/1.jpg'
preprocessed_data = preprocess_data(new_image_path)

# Pass preprocessed data to the model for prediction
prediction = loaded_model.predict([preprocessed_data])
print("Predicted class:", prediction)



