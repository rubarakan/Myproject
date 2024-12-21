import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.image import resize
import cv2

# Load the trained model
MODEL_PATH = "model-4.keras"  # Path to your trained model
model = load_model(MODEL_PATH)

# Sliding window function for patch extraction
def sliding_window(image, patch_size=128, stride=64):
    patches = []
    positions = []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size, :]
            patches.append(patch)
            positions.append((x, y))
    return np.array(patches), positions

# Function to reconstruct image from patches
def reconstruct_image(patches, positions, image_shape, patch_size=128, stride=64):
    h, w = image_shape[:2]
    reconstructed = np.zeros((h, w))
    count = np.zeros((h, w))

    for patch, (x, y) in zip(patches, positions):
        reconstructed[y:y + patch_size, x:x + patch_size] += patch.squeeze()
        count[y:y + patch_size, x:x + patch_size] += 1

    return reconstructed / np.maximum(count, 1)

# Function to predict road segmentation
def predict_segmentation(image, patch_size=128, stride=64, threshold=0.5):
    img_resized = resize(image, (256, 256)).numpy()  # Resize image if necessary
    patches, positions = sliding_window(img_resized, patch_size, stride)
    predictions = model.predict(patches, verbose=0)
    reconstructed = reconstruct_image(predictions, positions, img_resized.shape)
    binary_mask = (reconstructed >= threshold).astype(np.uint8)  # Thresholding for binary mask
    return binary_mask

# Streamlit UI
st.title("Road Segmentation using Deep Learning")
st.write("Upload an image, and the model will segment the roads.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the uploaded image
    st.subheader("Original Image")
    file_bytes = uploaded_file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the original image
    st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

    # Preprocess and segment the image
    st.subheader("Segmenting...")
    segmented_mask = predict_segmentation(image_rgb)

    # Display the segmentation result
    st.subheader("Segmented Image")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image_rgb)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(segmented_mask, cmap="gray")
    ax[1].set_title("Segmented Roads")
    ax[1].axis("off")

    st.pyplot(fig)

    # Option to download the segmented mask
    segmented_mask_uint8 = (segmented_mask * 255).astype(np.uint8)  # Convert to uint8
    success, buffer = cv2.imencode(".png", segmented_mask_uint8)
    if success:
        st.download_button(label="Download Segmented Image", 
                           data=buffer.tobytes(), 
                           file_name="segmented_result.png", 
                           mime="image/png")
