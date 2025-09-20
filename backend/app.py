from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from huggingface_hub import hf_hub_download

# Hugging Face repo where your models are stored
HF_REPO_ID = "Iamashwin1903/mango-disease"

# Model filenames in Hugging Face
MODELS = {
    "model1": "resnet.h5",
    "model2": "vgg-16.h5",
    "model3": "mobileVnet.h5",
    "model4": "inceptionV3.h5",
}

# Labels in model output order
LABELS = ["Healthy", "Anthracnose"]

# Initialize FastAPI app
app = FastAPI()

# ✅ Proper CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mango-disease-ui.onrender.com"],  # frontend Render URL
    allow_credentials=True,
    allow_methods=["*"],   # allow all HTTP methods
    allow_headers=["*"],   # allow all headers
)

# Cache for loaded models
model_cache = {}

def load_model_from_hf(model_filename: str):
    """Download and load a model from Hugging Face (with caching)."""
    if model_filename in model_cache:
        return model_cache[model_filename]

    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=model_filename)
        model = tf.keras.models.load_model(model_path)
        model_cache[model_filename] = model
        return model
    except Exception as e:
        print(f"Error loading model {model_filename} from Hugging Face: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model {model_filename}")

def preprocess_image(image, target_size):
    """Resize and normalize the image for prediction."""
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32")
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def home():
    return {"message": "Mango Disease API is running!"}

@app.post("/predict")
async def predict_disease(model_name: str, file: UploadFile = File(...)):
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Load the chosen model
        model_filename = MODELS[model_name]
        model = load_model_from_hf(model_filename)

        # Get model input shape
        model_input_shape = model.input_shape
        target_size = (model_input_shape[1], model_input_shape[2])

        # Preprocess and predict
        preprocessed_image = preprocess_image(image, target_size=target_size)
        prediction = model.predict(preprocessed_image)[0]

        healthy_prob = float(prediction[0])
        anthracnose_prob = 1.0 - healthy_prob  # simple 2-class assumption

        results = {LABELS[0]: healthy_prob, LABELS[1]: anthracnose_prob}
        return {"predictions": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
