import streamlit as st

import cv2
import numpy as np
from tf_explain.core.grad_cam import GradCAM
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

@st.cache(allow_output_mutation=True)
def load_cnn_model():
    cnn_model = load_model('covid19.model')

    return cnn_model

model = load_cnn_model()

st.write('# Covid-19 Pneumonia Classifier')
st.write('This Deep learning framework is for the purpose of computer aided diagnosis of covid-19 from chest x-ray images ')
st.write('Select an image from the left pane and leave the rest to the neural network...')


uploaded_image = st.sidebar.file_uploader("Choose a JPG file", type="jpeg")
if uploaded_image:
    st.sidebar.info('Uploaded image:')
    st.sidebar.image(uploaded_image, width=240)
    grad_cam_button = st.sidebar.button('Grad CAM')
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (244, 244))
    image = img_to_array(image)
    image1 = image.copy()
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    
    covid = yhat[0][0]
    normal = yhat[0][1]
    
    if covid > normal:
        classes = 'covid' 
    else:
        classes = 'normal' 
   
                
    x = max(yhat)
    x = max(x)
    
    if classes == 'covid':
        y = 0
    else:
        y = 1
    
    
    
    st.subheader(classes + ':  ' + str(x))
    
    if grad_cam_button:
        data = ([image1], None)
        explainer = GradCAM()
        grad_cam_grid = explainer.explain(
            data, model, class_index=y, layer_name="block5_conv3"
        )
        st.image(grad_cam_grid, width = 300)