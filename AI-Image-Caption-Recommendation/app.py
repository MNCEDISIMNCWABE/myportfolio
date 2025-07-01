from flask import Flask, request, jsonify
from flasgger import Swagger
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
Swagger(app)

# Load CLIP model once at startup
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Predefined candidate captions
CANDIDATE_CAPTIONS = [
"Confidence in every detail.",
"Professional excellence personified.",
"Dressed for success.",
"The face of determination.",
"Ambition meets preparation.",
"Leadership starts with presence.",
"Making an impression that lasts.",
"Where professionalism meets personality.",
"Success begins with presentation.",
"The power of professional presence.",
"Ready for whatever business brings.",
"Excellence in every interaction.",
"Representing with distinction.",
"The art of professional poise.",
"Beyond the business attire.",
"Projecting confidence and capability.",
"Where preparation meets opportunity.",
"The standard of professionalism.",
"Elevating the professional experience.",
"Crafting a professional legacy.",
"The intersection of style and substance.",
"Poised for professional excellence.",
"The language of professional presence.",
"Where ambition meets action.",
"The blueprint for professional success."]

@app.route('/caption-image', methods=['POST'])
def caption_image():
    """
    Generate recommended captions for an uploaded image
    ---
    tags:
      - Image Captioning
    consumes:
      - multipart/form-data
    parameters:
      - name: image
        in: formData
        type: file
        required: true
        description: The image to caption
      - name: top_n
        in: query
        type: integer
        default: 5
        description: Number of top captions to return
    responses:
      200:
        description: List of recommended captions with similarity scores
        schema:
          id: caption_response
          properties:
            captions:
              type: array
              items:
                type: object
                properties:
                  caption:
                    type: string
                  similarity:
                    type: number
      400:
        description: Error processing request
    """
    try:
        # Get image file from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        top_n = int(request.args.get('top_n', 5))
        
        # Process image
        image = Image.open(BytesIO(image_file.read())).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True)
        
        # Generate embeddings
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            text_inputs = processor(text=CANDIDATE_CAPTIONS, return_tensors="pt", padding=True)
            text_features = model.get_text_features(**text_inputs)
        
        # Calculate similarities
        image_features = image_features.detach().cpu().numpy()
        text_features = text_features.detach().cpu().numpy()
        similarities = cosine_similarity(image_features, text_features)[0]
        
        # Get top captions
        ranked_indices = np.argsort(similarities)[::-1]
        results = [{
            'caption': CANDIDATE_CAPTIONS[i],
            'similarity': float(similarities[i])
        } for i in ranked_indices[:top_n]]
        
        return jsonify({'captions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return """
    <h1>Image Caption Recommendation API</h1>
    <p>Visit <a href="/apidocs">/apidocs</a> for Swagger UI</p>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)