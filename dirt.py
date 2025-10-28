import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io
from streamlit_cropper import st_cropper

# Page configuration
st.set_page_config(page_title="Dirt Analyzer Dashboard", layout="wide")
st.title("Dirt Comparison Dashboard with Cropping & Color Analysis")

# Sidebar uploads
st.sidebar.header("Upload Images")
reference_file = st.sidebar.file_uploader("Upload Clean Reference Image", type=["jpg", "png"])
uploaded_files = st.sidebar.file_uploader("Upload Sample Images", type=["jpg", "png"], accept_multiple_files=True)

# Initialize session state
if "cropped_reference" not in st.session_state:
    st.session_state.cropped_reference = None
if "cropped_samples" not in st.session_state:
    st.session_state.cropped_samples = {}

# Color analysis function
def analyze_color(image):
    img_array = np.array(image)
    avg_color = img_array.mean(axis=(0, 1))  # [R, G, B]
    return [round(c, 2) for c in avg_color]

if reference_file and uploaded_files:
    st.sidebar.success("Files uploaded successfully!")

    # Crop reference image
    if st.session_state.cropped_reference is None:
        st.write("### Crop Reference Image")
        ref_image = Image.open(reference_file)
        cropped_ref = st_cropper(ref_image, realtime_update=True, box_color="blue")
        st.image(cropped_ref, caption="Selected Reference Region", width=250)
        if st.button("Save Reference Crop"):
            st.session_state.cropped_reference = cropped_ref
            st.success("Reference crop saved!")
    else:
        st.write("✅ Reference crop saved.")
        st.image(st.session_state.cropped_reference, caption="Reference Crop", width=250)

    # Crop sample images
    if st.session_state.cropped_reference:
        st.write("### Crop Sample Images")
        sample_names = [f"Sample {i+1}" for i in range(len(uploaded_files))]
        selected_sample = st.selectbox("Select a sample to crop", sample_names)
        selected_index = sample_names.index(selected_sample)

        if selected_sample in st.session_state.cropped_samples:
            st.write(f"✅ Crop already saved for {selected_sample}")
            st.image(st.session_state.cropped_samples[selected_sample], caption=f"Saved Crop for {selected_sample}", width=250)
        else:
            image = Image.open(uploaded_files[selected_index])
            cropped_img = st_cropper(image, realtime_update=True, box_color="green")
            st.image(cropped_img, caption=f"Selected Region for {selected_sample}", width=250)
            if st.button("Save Sample Crop"):
                st.session_state.cropped_samples[selected_sample] = cropped_img
                st.success(f"Crop saved for {selected_sample}")

    # Analyze when all crops are saved
    if len(st.session_state.cropped_samples) == len(uploaded_files):
        if st.button("Analyze Dirt"):
            # Reference metrics
            ref_gray = cv2.cvtColor(np.array(st.session_state.cropped_reference), cv2.COLOR_RGB2GRAY)
            ref_score = 255 - np.mean(ref_gray)
            ref_color = analyze_color(st.session_state.cropped_reference)

            results = []
            for sample_name, cropped_img in st.session_state.cropped_samples.items():
                gray = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2GRAY)
                dirt_score = 255 - np.mean(gray)
                normalized = ((dirt_score - ref_score) / ref_score) * 100

                avg_color = analyze_color(cropped_img)
                color_diff = sum(abs(np.array(avg_color) - np.array(ref_color)))  # Simple RGB diff

                results.append({
                    "Sample": sample_name,
                    "Dirt Score": round(dirt_score, 2),
                    "Normalized (%)": round(normalized, 2),
                    "Avg Color (R,G,B)": avg_color,
                    "Color Diff": round(color_diff, 2)
                })

            # Organized layout: 3 columns for samples
            st.write("### Dirt Analysis Results")
            cols = st.columns(3)
            for idx, row in enumerate(results):
                color_rgb = row["Avg Color (R,G,B)"]
                color_hex = '#%02x%02x%02x' % (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))
                with cols[idx % 3]:
                    st.image(st.session_state.cropped_samples[row["Sample"]], caption=row["Sample"], width=200)
                    st.markdown(
                        f"""
                        <div style="margin-top:8px;">
                            <div style="width:60px;height:60px;background-color:{color_hex};border:2px solid #000;margin-bottom:8px;"></div>
                            <span style="font-size:14px;">
                                Dirt: {row['Dirt Score']} | Norm: {row['Normalized (%)']}%<br>
                                Color Diff: {row['Color Diff']}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Charts
            df = pd.DataFrame(results)
            st.write("#### Dirt Score Comparison")
            st.bar_chart(df.set_index("Sample")[["Dirt Score"]])

            # Download CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button("Download Results as CSV", csv_buffer.getvalue(), "dirt_analysis.csv", "text/csv")
