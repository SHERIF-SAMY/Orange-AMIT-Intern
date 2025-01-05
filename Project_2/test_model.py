# import numpy as np
# import tensorflow as tf

# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt

# def predict_image_class(image_path):
#     model = tf.keras.models.load_model('model_fer.keras')

#     img = image.load_img(image_path, target_size=(48, 48), color_mode="grayscale")
#     img = np.array(img)
#     plt.imshow(img, cmap='gray')

#     label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

#     img = np.expand_dims(img, axis=0)
#     img = img.reshape(1, 48, 48, 1)
#     result = model.predict(img)
#     result = list(result[0])

#     img_index = result.index(max(result))
#     predicted_class = label_dict[img_index]
#     plt.show()
#     return predicted_class

# # Example usage
# predicted_class = predict_image_class(r"C:\Users\Asus\Desktop\test.png")
# print(f"Predicted class: {predicted_class}")
#############################################################################

# import streamlit as st
# from PIL import Image
# import numpy as np
# from tensorflow.keras.models import load_model

# # Assuming the pre-trained model is saved as "emotion_model.h5"

# EMOTION_LABELS = ["Angry", "Fear", "Surprise","Neutral", "Sad","Happy" ]

# def preprocess_image(image):
#     """
#     Preprocess the uploaded image for prediction.
#     - Convert to grayscale
#     - Resize to 48x48
#     - Normalize pixel values
#     """
#     image = image.resize((48, 48))
#     image_array = np.array(image) / 255.0  # Normalize pixel values
#     if len(image_array.shape) == 2:  # Ensure the image is grayscale
#         image_array = np.expand_dims(image_array, axis=-1)
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# def predict_emotion(model, image_array):
#     """Predict the emotion for the preprocessed image."""
#     predictions = model.predict(image_array)
#     predicted_class = np.argmax(predictions, axis=1)[0]
#     return EMOTION_LABELS[predicted_class]

# st.title("Emotion Recognition App")
# st.write("Upload an image to recognize the emotion.")
# model = load_model('model_fer.keras')
# # model = load_model('model_fer2013_updated.h5')
# # import tensorflow as tf
# # new_model = tf.keras.models.load_model('model_fer.keras')

# # TypeError: Error when deserializing class 'InputLayer' using config={'batch_shape': [None, 128, 173, 1], 'dtype': 'float32', 'sparse': False, 'name': 'input_layer_1'}.
# # Exception encountered: Unrecognized keyword arguments: ['batch_shape']


#     # Upload image section
# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#         # Display the uploaded image
#         image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         # Preprocess the image and make prediction
#         if st.button("Predict Emotion"):
#             processed_image = preprocess_image(image)
#             emotion = predict_emotion(model, processed_image)
#             st.write(f"Predicted Emotion: **{emotion}**")

##################################################################3
import tensorflow as tf
import pickle

# Step 1: Load the .keras model
model = tf.keras.models.load_model("model_fer.keras")

# Step 2: Save the model as a .pkl file
with open("model_fer.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model has been successfully saved as model.pkl")