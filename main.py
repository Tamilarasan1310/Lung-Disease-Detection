import streamlit as st
import tensorflow as tf
import numpy as np

# ------------------ Model Prediction ------------------
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# ------------------ Sidebar ------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Select Page",
    ["Home", "About", "Lung Disease Detection"]
)

# ------------------ Home Page ------------------
if app_mode == "Home":
    st.header("LUNG DISEASE DETECTION SYSTEM")
    st.image("home_page.jpg", use_column_width=True)

    st.markdown("""
    Welcome to the **Lung Disease Detection System** 🫁  

    This system helps in identifying lung diseases from medical images using deep learning.

    ### How It Works
    1. **Upload Image:** Upload a chest X-ray image.
    2. **Analysis:** The model analyzes the image.
    3. **Prediction:** The detected lung condition is displayed.

    ### Why Use This System?
    - High accuracy using deep learning
    - Simple and user-friendly interface
    - Fast predictions

    👉 Go to **Lung Disease Detection** from the sidebar to get started.
    """)

# ------------------ About Page ------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### Dataset Information
    The dataset consists of labeled chest X-ray images classified into five categories:
    - Bacterial Pneumonia  
    - Corona Virus Disease  
    - Normal  
    - Tuberculosis  
    - Viral Pneumonia  

    The dataset was split into training, validation, and testing sets.
    """)

# ------------------ Prediction Page ------------------
elif app_mode == "Lung Disease Detection":
    st.header("Lung Disease Detection")

    test_image = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                result_index = model_prediction(test_image)

                class_name = [
                    'Bacterial Pneumonia',
                    'Corona Virus Disease',
                    'Normal',
                    'Tuberculosis',
                    'Viral Pneumonia'
                ]

                st.success(f"🩺 Prediction: **{class_name[result_index]}**")
