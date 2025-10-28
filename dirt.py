import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from streamlit_cropper import st_cropper

st.set_page_config(page_title="Dirt Analyzer Dashboard", layout="wide")
st.title("Dirt Comparison Dashboard with Cropping")

# Sidebar for uploads
st.sidebar.header("Upload Images")
reference_file = st.sidebar.file_uploader("Upload Clean Reference Image", type=["jpg", "png"])
uploaded_files = st.sidebar.file_uploader("Upload Sample Images", type=["jpg", "png"], accept_multiple_files=True)

if reference_file and uploaded_files:
    st.sidebar.success("Files uploaded successfully!")
    st.write("### Enter names for each sample:")
    sample_names = []
    for i in range(len(uploaded_files)):
        name = st.text_input(f"Name for Sample {i+1}", f"Sample {i+1}")
        sample_names.append(name)

    if st.button("Analyze Dirt"):
        # Crop reference image
        st.write("#### Crop Reference Image")
        ref_image = Image.open(reference_file)
        cropped_ref = st_cropper(ref_image, realtime_update=True, box_color="blue")
        st.image(cropped_ref, caption="Selected Reference Region", use_column_width=True)

        # Calculate reference dirt score
        ref_gray = cv2.cvtColor(np.array(cropped_ref), cv2.COLOR_RGB2GRAY)
        ref_score = 255 - np.mean(ref_gray)

        results = []
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(ref_image, caption="Original Reference", use_column_width=True)

        for idx, uploaded_file in enumerate(uploaded_files):
            st.write(f"#### Crop Region for {sample_names[idx]}")
            image = Image.open(uploaded_file)
            cropped_img = st_cropper(image, realtime_update=True, box_color="green")
            st.image(cropped_img, caption=f"Selected Region for {sample_names[idx]}", use_column_width=True)

            gray = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2GRAY)
            dirt_score = 255 - np.mean(gray)
            normalized = ((dirt_score - ref_score) / ref_score) * 100
            results.append({
                "Sample": sample_names[idx],
                "Dirt Score": round(dirt_score, 2),
                "Normalized (%)": round(normalized, 2)
            })

        # Display results
        df = pd.DataFrame(results)
        with col2:
            st.write("### Dirt Scores")
            st.dataframe(df)
st.bar_chart(df.set_index("Sample")[["Normalized (%)"]])
