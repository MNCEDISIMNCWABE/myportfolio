from flask import Flask, request, jsonify
from flasgger import Swagger
import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Initialize Flask app
app = Flask(__name__)
swagger = Swagger(app)  # Initialize Swagger

# Load the model and scaler
model_path = 'potholes_predictor_model.pkl'
scaler_path = 'potholes_image_scaler.pkl'
model, scaler = joblib.load(model_path), joblib.load(scaler_path)

# Define a directory for uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_single_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    flattened_image = img_array.reshape(1, -1)
    return flattened_image

def predict_single_image(img_path, model, scaler):
    img_array = preprocess_single_image(img_path)
    img_array = scaler.transform(img_array)
    likelihood = model.predict_proba(img_array)[:, 1][0]
    return likelihood

@app.route('/predict', methods=['POST'])
def predict():
    """
    Upload an image and get the likelihood of it containing a pothole.
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: The image file to upload and predict.
    responses:
      200:
        description: The likelihood of the image containing a pothole.
        schema:
          type: object
          properties:
            likelihood:
              type: float
              description: The predicted likelihood (between 0 and 1).
      400:
        description: Bad request, if the file is not provided.
        schema:
          type: object
          properties:
            error:
              type: string
              description: Error message.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Predict the likelihood of the image containing a pothole
        likelihood = predict_single_image(file_path, model, scaler)
        
        # Optionally, delete the file after prediction
        os.remove(file_path)
        
        return jsonify({"likelihood": likelihood})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
