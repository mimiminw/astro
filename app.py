import streamlit as st
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Streamlit 페이지 설정
st.set_page_config(page_title="Galaxy FITS Analyzer", layout="wide")

# 제목
st.title("Galaxy FITS File Analyzer")
st.write("Upload a FITS file to visualize the galaxy image and analyze its properties.")

# 파일 업로더
uploaded_file = st.file_uploader("Choose a FITS file", type=["fits", "fit", "fz"])

def analyze_fits_file(file):
    """FITS 파일을 분석하고 정보를 추출하는 함수"""
    try:
        # FITS 파일 읽기
        fits_data = fits.open(file)
        image_data = fits_data[0].data
        header = fits_data[0].header
        fits_data.close()

        # 기본 정보 추출
        brightness_mean = np.nanmean(image_data)
        brightness_std = np.nanstd(image_data)
        image_shape = image_data.shape
        exposure_time = header.get('EXPTIME', 'N/A')

        # 정보 표시
        st.subheader("Basic Information")
        st.write(f"**Image Dimensions**: {image_shape[0]} x {image_shape[1]} pixels")
        st.write(f"**Mean Brightness**: {brightness_mean:.2f}")
        st.write(f"**Brightness Standard Deviation**: {brightness_std:.2f}")
        st.write(f"**Exposure Time**: {exposure_time} seconds" if exposure_time != 'N/A' else "**Exposure Time**: Not available")

        # 이미지 시각화
        st.subheader("Galaxy Image")
        fig, ax = plt.subplots()
        ax.imshow(image_data, cmap='gray', origin='lower')
        ax.set_title("FITS Image")
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(ax.imshow(image_data, cmap='gray', origin='lower'), ax=ax, label="Intensity")
        
        # Matplotlib 이미지를 Streamlit에 표시
        buf = BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        st.markdown(f'<img src="data:image/png;base64,{data}" alt="FITS Image">', unsafe_allow_html=True)

        # 허블 분류 (단순화된 규칙 기반)
        st.subheader("Hubble Classification (Simplified)")
        aspect_ratio = image_shape[0] / image_shape[1]
        central_brightness = np.nanmean(image_data[
            int(image_shape[0]*0.4):int(image_shape[0]*0.6),
            int(image_shape[1]*0.4):int(image_shape[1]*0.6)
        ])
        outer_brightness = np.nanmean(np.concatenate([
            image_data[:int(image_shape[0]*0.1), :],
            image_data[int(image_shape[0]*0.9):, :],
            image_data[:, :int(image_shape[1]*0.1)],
            image_data[:, int(image_shape[1]*0.9):]
        ]))

        # 단순화된 분류 로직
        if aspect_ratio > 1.5 or aspect_ratio < 0.67:
            classification = "Elliptical (E)"
            description = "The galaxy appears elongated, suggesting an elliptical shape (E0-E7)."
        elif central_brightness > 2 * outer_brightness:
            classification = "Spiral (S)"
            description = "High central brightness indicates a possible spiral galaxy with a bright core."
        else:
            classification = "Irregular (Irr)"
            description = "The galaxy lacks clear structure, suggesting an irregular type."

        st.write(f"**Classification**: {classification}")
        st.write(f"**Description**: {description}")

    except Exception as e:
        st.error(f"Error processing FITS file: {str(e)}")

# 파일이 업로드되었을 때 분석 실행
if uploaded_file is not None:
    st.write("Processing FITS file...")
    analyze_fits_file(uploaded_file)
else:
    st.info("Please upload a FITS file to begin analysis.")

# 추가 정보
st.sidebar.header("About")
st.sidebar.write("This app analyzes FITS files to visualize astronomical images and classify galaxies based on a simplified Hubble classification system.")
st.sidebar.write("Note: The classification is a basic approximation and not a substitute for professional analysis.")
