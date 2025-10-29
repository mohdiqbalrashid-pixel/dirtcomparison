import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_cropper import st_cropper
import io

# Page configuration
st.set_page_config(page_title="Dirt Comparison Dashboard", layout="wide")
st.title("Dirt Comparison Dashboard - Heatmap Enhanced with Legend")

# Sidebar uploads
st.sidebar.header("Upload Images")
reference_file = st.sidebar.file_uploader("Upload Clean Reference Image", type=["jpg", "png"])
uploaded_files = st.sidebar.file_uploader("Upload Sample Images", type=["jpg", "png"], accept_multiple_files=True)

# Sidebar controls
heatmap_intensity = st.sidebar.slider("Heatmap Intensity", 0.0, 1.0, 0.6)
cmap_choice = st.sidebar.selectbox("Heatmap Style", ["JET", "HOT", "TURBO"])

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

# Heatmap function with inversion fix
def to_heatmap(image, intensity, cmap_choice):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # Invert grayscale so dirt = high intensity
    inverted = cv2.bitwise_not(gray_enhanced)

    # Normalize intensity
    normalized = cv2.normalize(inverted, None, 0, 255, cv2.NORM_MINMAX)

    # Apply selected color map
    cmap_dict = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "TURBO": cv2.COLORMAP_TURBO
    }
    heatmap = cv2.applyColorMap(normalized, cmap_dict[cmap_choice])

    # Blend with original based on intensity slider
    blended = cv2.addWeighted(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR), 1 - intensity, heatmap, intensity, 0)

    return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))

# Generate heatmap legend
def generate_legend(cmap_choice):
    gradient = np.linspace(0, 255, 256).astype(np.uint8)
    gradient = np.tile(gradient, (50, 1))  # 50px height
    cmap_dict = {
        "JET": cv2.COLORMAP_JET,
        "HOT": cv2.COLORMAP_HOT,
        "TURBO": cv2.COLORMAP_TURBO
    }
    color_bar = cv2.applyColorMap(gradient, cmap_dict[cmap_choice])
    return Image.fromarray(cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB))

# Color analysis function
def analyze_color(image):
    img_array = np.array(image)
    avg_color = img_array.mean(axis=(0, 1))  # [R, G, B]
    return [round(c, 2) for c in avg_color]

if reference_file and uploaded_files:
    st.sidebar.success("Files uploaded successfully!")

    # Split page into two columns
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
                # Show heatmap legend
                st.write("### Heatmap Legend")
                st.image(generate_legend(cmap_choice), caption="Blue = Clean | Red/Yellow = Dirtier", use_column_width=True)

                # Reference metrics
                ref_gray = cv2.cvtColor(np.array(st.session_state.cropped_reference), cv2.COLOR_RGB2GRAY)
                ref_score = 255 - np.mean(ref_gray)
                ref_color = analyze_color(st.session_state.cropped_reference)

                results = []
                results.append({
                    "Sample": "Reference",
                    "Dirt Score": round(ref_score, 2),
                    "Normalized (%)": 0.0,
                    "Avg Color (R,G,B)": ref_color,
                    "Color Diff": 0.0
                })

                for sample_name, cropped_img in st.session_state.cropped_samples.items():
                    gray = cv2.cvtColor(np.array(cropped_img), cv2.COLOR_RGB2GRAY)
                    dirt_score = 255 - np.mean(gray)
                    normalized = ((dirt_score - ref_score) / ref_score) * 100
                    avg_color = analyze_color(cropped_img)
                    color_diff = sum(abs(np.array(avg_color) - np.array(ref_color)))

                    results.append({
                        "Sample": sample_name,
                        "Dirt Score": round(dirt_score, 2),
                        "Normalized (%)": round(normalized, 2),
                        "Avg Color (R,G,B)": avg_color,
                        "Color Diff": round(color_diff, 2)
                    })

                # Display analysis with expandable zoom
                st.write("### Analysis Results with Adjustable Heatmap")
                for row in results:
                    st.markdown(f"**{row['Sample']}**")
                    img = st.session_state.cropped_reference if row['Sample'] == "Reference" else st.session_state.cropped_samples[row['Sample']]
                    
                    # Side-by-side thumbnails
                    col_img1, col_img2 = st.columns([1, 1])
                    with col_img1:
                        st.image(img, caption="Original", width=200)
                    with col_img2:
                        st.image(to_heatmap(img, heatmap_intensity, cmap_choice), caption=f"Heatmap ({cmap_choice})", width=200)
                    
                    # Expanders for zoomed view
                    with st.expander(f"View {row['Sample']} in full size"):
                        st.image(img, caption="Original Full Size", use_column_width=True)
                        st.image(to_heatmap(img, heatmap_intensity, cmap_choice), caption=f"Heatmap Full Size ({cmap_choice})", use_column_width=True)
                    
                    st.write(f"Dirt Score: {row['Dirt Score']} | Normalized: {row['Normalized (%)']}% | Color Diff: {row['Color Diff']}")
                    st.markdown("---")

                # Download CSV
                df = pd.DataFrame(results)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button("Download Analysis as CSV", csv_buffer.getvalue(), "dirt_analysis.csv", "text/csv")
