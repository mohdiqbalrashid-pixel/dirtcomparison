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

# Reset button
if st.sidebar.button("Reset All Crops"):
    st.session_state.cropped_reference = None
    st.session_state.cropped_samples = {}
    st.success("All crops have been reset!")

# Color analysis function
def analyze_color(image):
    img_array = np.array(image)
    avg_color = img_array.mean(axis=(0, 1))  # [R, G, B]
    return [round(c, 2) for c in avg_color]

if reference_file and uploaded_files:
    st.sidebar.success("Files uploaded successfully!")

    # Split page into two main columns
    col_left, col_right = st.columns([1, 2])

    # Left column: Cropping workflow
    with col_left:
        st.write("### Cropping Section")

        # Crop reference image
        if st.session_state.cropped_reference is None:
            st.write("#### Crop Reference Image")
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
            st.write("#### Crop Sample Images")
            sample_names = [f"Sample {i+1}" for i in range(len(uploaded_files))]
            selected_sample = st.selectbox("Select a sample to crop or edit", sample_names)
            selected_index = sample_names.index(selected_sample)

            if selected_sample in st.session_state.cropped_samples:
                st.write(f"✅ Crop saved for {selected_sample}")
                st.image(st.session_state.cropped_samples[selected_sample], caption=f"Saved Crop for {selected_sample}", width=250)
                if st.button("Edit Crop"):
                    cropped_img = st_cropper(Image.open(uploaded_files[selected_index]), realtime_update=True, box_color="orange")
                    st.image(cropped_img, caption=f"Editing Crop for {selected_sample}", width=250)
                    if st.button("Save Edited Crop"):
                        st.session_state.cropped_samples[selected_sample] = cropped_img
                        st.success(f"Crop updated for {selected_sample}")
            else:
                image = Image.open(uploaded_files[selected_index])
                cropped_img = st_cropper(image, realtime_update=True, box_color="green")
                st.image(cropped_img, caption=f"Selected Region for {selected_sample}", width=250)
                if st.button("Save Sample Crop"):
                    st.session_state.cropped_samples[selected_sample] = cropped_img
                    st.success(f"Crop saved for {selected_sample}")

    # Right column: Analysis section
    with col_right:
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

                # Split right column into two sub-columns
                col_r1, col_r2 = st.columns([1, 1])

                # Display color swatches and metrics in col_r1
                with col_r1:
                    st.write("### Dirt Analysis Results")
                    for row in results:
                        color_rgb = row["Avg Color (R,G,B)"]
                        color_hex = '#%02x%02x%02x' % (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))
                        st.markdown(
                            f"""
                            <div style="display:flex;align-items:center;margin-bottom:16px;">
                                <div style="width:60px;height:60px;background-color:{color_hex};border:2px solid #000;margin-right:16px;"></div>
                                <span style="font-size:16px;">
                                    <b>{row['Sample']}</b><br>
                                    Dirt: {row['Dirt Score']} | Norm: {row['Normalized (%)']}%<br>
                                    Color Diff: {row['Color Diff']}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # Display charts and download in col_r2
                with col_r2:
                    df = pd.DataFrame(results)
                    st.write("#### Dirt Score Comparison")
                    st.bar_chart(df.set_index("Sample")[["Dirt Score"]])

                    st.write("#### Color Difference Comparison")
                    st.bar_chart(df.set_index("Sample")[["Color Diff"]])

                    # Download CSV
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    st.download_button("Download Results as CSV", csv_buffer.getvalue(), "dirt_analysis.csv", "text/csv")
