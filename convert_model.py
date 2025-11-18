import tensorflow as tf

# Load the newly trained model
model = tf.keras.models.load_model('posture_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the converted model
with open('posture_model.tflite', 'wb') as f:
    f.write(tflite_model)

print('Model converted to TensorFlow Lite format and saved as posture_model.tflite')
