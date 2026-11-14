
import streamlit as st
import requests
import base64
from PIL import Image, ImageOps
from io import BytesIO

st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Industrial Defect Detection System")
st.caption("Bottle defect detection using a lightweight PatchCore-like model.")

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader(
    "Upload a bottle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Analyze Image", type="primary"):

        with st.spinner("Analyzing image..."):

            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=120
                )

                response.raise_for_status()
                result = response.json()

                st.success("Analysis completed!")

                col1, col2, col3 = st.columns(3)

                col1.metric("Prediction", result["prediction"])
                col2.metric(
                    "Anomaly Score",
                    f'{result["anomaly_score"]:.4f}'
                )
                col3.metric(
                    "Provisional Threshold",
                    f'{result["threshold"]:.2f}'
                )

                heatmap_data = base64.b64decode(
                    result["heatmap_base64"]
                )

                heatmap = Image.open(
                    BytesIO(heatmap_data)
                ).convert("L")

                heatmap = heatmap.resize(image.size)

                colored_heatmap = ImageOps.colorize(
                    heatmap,
                    black="#000000",
                    white="#ff0000"
                )

                overlay = Image.blend(
                    image,
                    colored_heatmap,
                    alpha=0.35
                )

                st.subheader("Visual Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Original Image")
                    st.image(image, use_container_width=True)

                with col2:
                    st.write("Anomaly Heatmap")
                    st.image(
                        colored_heatmap,
                        use_container_width=True
                    )

                st.subheader("Heatmap Overlay")
                st.image(overlay, use_container_width=True)

                st.info(
                    "Heatmap shows anomaly intensity, not a confirmed "
                    "defect boundary. The threshold is provisional."
                )

            except requests.exceptions.RequestException as e:
                st.error(f"Backend API error: {e}")

            except Exception as e:
                st.error(f"Error: {e}")
