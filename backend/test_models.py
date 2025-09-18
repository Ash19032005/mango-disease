# import tensorflow as tf
# model3='models\mango-disease-mobileVnet.h5'
# try:
#     model = tf.keras.models.load_model(model3)
#     print("model3 loaded successfully!")
#     print("Model input shape:", model.input_shape)
# except Exception as e:
#     print(f"Error loading model3: {e}")

MODELS = {
    "model1": {
        "model": tf.keras.models.load_model('models/mango-disease-resnet.h5'),
        "preprocess": preprocess_resnet
    },
    "model2": {
        "model": tf.keras.models.load_model('models/mango-disease-vgg16.h5'),
        "preprocess": preprocess_vgg
    },
    "model3": {
        "model": tf.keras.models.load_model('models/mango-disease-mobileVnet.h5'),
        "preprocess": preprocess_mobilenet
    },
    "model4": {
        "model": tf.keras.models.load_model('models/mango-disease-InceptionV3.h5'),
        "preprocess": preprocess_inception
    }
}