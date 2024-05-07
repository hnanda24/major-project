from flask import Flask, jsonify, request
import base64
import pandas as pd
from arrow import now
from glob import glob
from img2vec_pytorch import Img2Vec
from io import BytesIO
from os.path import basename
from PIL import Image
import pickle

app = Flask(__name__)

# Define global variables
GLOB = 'Aero-engine_defect-detect_new/images'
SIZE = 512
STOP = 500
THUMBNAIL_SIZE = (100, 100)
model = Img2Vec(cuda=False, model='resnet-18')

# Helper functions
def embed(filename: str):
    with Image.open(filename) as image:
        return model.get_vec(image, tensor=True).numpy().reshape(SIZE,)

def png(filename: str) -> str:
    with Image.open(filename) as image:
        buffer = BytesIO()
        image.resize(THUMBNAIL_SIZE).save(buffer, format='png')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

def get_picture_from_glob(arg: str, tag: str, stop: int) -> list:
    time_get = now()
    result = [pd.Series(data=[tag, basename(input_file), embed(input_file), png(input_file)], index=['tag', 'name', 'value', 'image'])
        for index, input_file in enumerate(glob(arg)) if index < stop]
    print('encoded {} data {} rows in {}'.format(tag, len(result), now() - time_get))
    return result

# Define API endpoints
@app.route('/process_data', methods=['GET'])
def process_data():
    time_start = now()
    data_dict = {basename(folder): folder + '/*.jpg' for folder in glob(GLOB + '/*')}
    df = pd.DataFrame(flatten([get_picture_from_glob(value, key, STOP) for key, value in data_dict.items()]))
    df['label name'] = df['name'].apply(lambda x: x.replace('.jpg', '.txt'))
    print('done in {}'.format(now() - time_start))
    return jsonify({'message': 'Data processed successfully'})

@app.route('/train_model', methods=['GET'])
def train_model():
    # Your model training code here
    return jsonify({'message': 'Model trained successfully'})

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model from the file
    with open('random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Extract data from request (assuming it's a list of image file paths)
    data = request.json  # Assuming request contains JSON data with list of image file paths

    # Preprocess the data (extract embeddings)
    embeddings = [embed(filename) for filename in data]

    # Make predictions using the loaded model
    predictions = loaded_model.predict(embeddings)

    # Return predictions as JSON
    return jsonify({'predictions': predictions})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
