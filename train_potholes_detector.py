# train_and_save_model.py

import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def read_data(path_to_csv_file):
    return pd.read_csv(path_to_csv_file)

def preprocess_labels_and_ids(df):
    df['Label'] = df['Label'].astype(str)
    df['Image_ID'] = df['Image_ID'] + '.JPG'
    return df

def create_generators(train_df, image_dir, target_size=(32, 32), batch_size=5):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="Image_ID",
        y_col="Label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=False
    )
    
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=image_dir,
        x_col="Image_ID",
        y_col="Label",
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def preprocess_images(generator):
    images = []
    labels = []
    for _ in range(len(generator)):
        batch_images, batch_labels = next(generator)
        flattened_images = batch_images.reshape(batch_images.shape[0], -1)
        images.append(flattened_images)
        labels.append(batch_labels)
    return np.vstack(images), np.hstack(labels)

def standardize_data(X_train, X_val):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    return X_train, X_val, scaler

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=16,
        criterion='gini',
        class_weight='balanced',
        min_samples_split=10,
        max_leaf_nodes=20,
        max_features=0.5,
        bootstrap=True
    )
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_train, y_train, X_val, y_val):
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    
    train_accuracy = accuracy_score(y_train, (y_train_pred > 0.5).astype(int))
    val_accuracy = accuracy_score(y_val, (y_val_pred > 0.5).astype(int))
    
    return train_auc, val_auc, train_accuracy, val_accuracy

def save_model_and_scaler(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

def run_pipeline_and_save_model(train_df, image_dir, model_path, scaler_path):
    train_df = preprocess_labels_and_ids(train_df)
    
    train_generator, validation_generator = create_generators(train_df, image_dir)
    
    X_train, y_train = preprocess_images(train_generator)
    X_val, y_val = preprocess_images(validation_generator)
    
    X_train, X_val, scaler = standardize_data(X_train, X_val)
    
    model = train_random_forest(X_train, y_train)
    
    train_auc, val_auc, train_accuracy, val_accuracy = evaluate_model(model, X_train, y_train, X_val, y_val)
    
    print(f"Training AUC: {train_auc:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    save_model_and_scaler(model, scaler, model_path, scaler_path)


if __name__ == "__main__":
    image_dir = '/Users/mncedisimncwabe/Downloads/Potholes Detection/all_data'
    train_df = read_data('/Users/mncedisimncwabe/Downloads/Potholes Detection/train_ids_labels.csv')
    model_path = 'potholes_predictor_model.pkl'
    scaler_path = 'potholes_image_scaler.pkl'
    
    # Train and save the model
    run_pipeline_and_save_model(train_df, image_dir, model_path, scaler_path)