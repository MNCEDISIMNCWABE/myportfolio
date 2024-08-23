import joblib
import numpy as np
from tensorflow.keras.preprocessing import image

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

def load_model_and_scaler(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model loaded from {model_path}")
    print(f"Scaler loaded from {scaler_path}")
    return model, scaler

def predict_uploaded_image(img_path, model_path, scaler_path):
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    likelihood = predict_single_image(img_path, model, scaler)
    print(f"Predicted Likelihood of Pothole (%): {likelihood:.4f}")


if __name__ == "__main__":
    model_path = 'potholes_predictor_model.pkl'
    scaler_path = 'potholes_image_scaler.pkl'
    
    # Predict on a single image using the saved model
    img_path = '/Users/mncedisimncwabe/Downloads/Potholes Detection/all_data/fLqXIwUZjcaiyxZ.JPG'
    predict_uploaded_image(img_path, model_path, scaler_path)
