import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io

st.title("Dirt Comparison Tool")

uploaded_files = st.file_uploader("Upload sample images", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write("Enter names for each sample:")
    sample_names = []
    for i in range(len(uploaded_files)):
        name = st.text_input(f"Name for Sample {i+1}", f"Sample {i+1}")
        sample_names.append(name)

    if st.button("Analyze Dirt"):
        results = []
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            avg_intensity = np.mean(gray)
            dirt_score = 255 - avg_intensity
            results.append({"Sample": sample_names[idx], "Dirt Score": round(dirt_score, 2)})

            st.image(image, caption=f"{sample_names[idx]}", use_container_width=True)

        # Display table
        df = pd.DataFrame(results)
        st.write("Dirt Scores:")
        st.dataframe(df)

        # Bar chart
        st.bar_chart(df.set_index("Sample"))

        # Download CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("Download Results as CSV", csv_buffer.getvalue(), "dirt_scores.csv", "text/csv")
