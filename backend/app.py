from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# ---------------------------
# FastAPI app setup
# ---------------------------
app = FastAPI(title="Mango Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load TFLite models
# ---------------------------
MODEL_PATHS = {
    "model1": "models/mango-disease-resnet_quantized.tflite",
    "model2": "models/mango-disease-vgg16_quantized.tflite",
    "model3":"models/mango-disease-mobileVnet_quantized.tflite",
    "model4":"models/mango-disease-InceptionV3_quantized.tflite"
}

interpreters = {}
input_details = {}
output_details = {}

for name, path in MODEL_PATHS.items():
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    interpreters[name] = interpreter
    input_details[name] = interpreter.get_input_details()
    output_details[name] = interpreter.get_output_details()

# ---------------------------
# Preprocess uploaded image
# ---------------------------
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Resize, normalize, and expand dims for TFLite model input
    """
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32)

    # Ensure 3 channels
    if img_array.shape[-1] == 4:  # RGBA to RGB
        img_array = img_array[..., :3]
    elif img_array.ndim == 2:  # grayscale to RGB
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Normalize if model expects float input (0-1)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

# ---------------------------
# Prediction function
# ---------------------------
def predict_tflite(image: Image.Image, model_name: str):
    if model_name not in interpreters:
        raise ValueError(f"Model '{model_name}' not found.")

    interpreter = interpreters[model_name]
    input_index = input_details[model_name][0]['index']
    output_index = output_details[model_name][0]['index']

    img = preprocess_image(image)
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)[0]

    # Handle single-output (sigmoid) or two-output (softmax)
    if len(output_data) == 1:
        anthracnose_prob = float(output_data[0])
        healthy_prob = 1.0 - anthracnose_prob
        predictions = {
            "Anthracnose":healthy_prob,
            "Healthy":  anthracnose_prob
        }
    elif len(output_data) == 2:
        predictions = {
            "Anthracnose": float(output_data[1]),
            "Healthy": float(output_data[0])
        }
    else:
        raise ValueError(f"Unexpected model output shape: {output_data.shape}")

    return predictions


# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Query("model1")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        predictions = predict_tflite(image, model_name)
        return {"predictions": predictions}
    except Exception as e:
        return {"error": str(e)}
