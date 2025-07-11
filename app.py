import streamlit as st
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from datetime import datetime
import matplotlib.pyplot as plt
import tempfile
import os
import fitsio
from io import BytesIO
import pandas as pd
from fpdf import FPDF

st.set_page_config(page_title="Galaxy FITS Analyzer", layout="wide")
st.title("🌌 Galaxy FITS File Analyzer & Report Generator")

uploaded_file = st.file_uploader("Choose a FITS file", type=["fits", "fit", "fz"])

seoul_location = EarthLocation(lat=37.5665, lon=126.9780, height=50)
now = Time(datetime.utcnow())

def load_fits_data(file, filename):
    ext = filename.lower().split('.')[-1]
    hdu_list = []

    if ext == 'fz':
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fz") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            with fitsio.FITS(tmp_path) as f:
                for i in range(len(f)):
                    hdu_list.append((i, f[i].get_extname() or f"HDU {i}"))
            return f, 'fz', tmp_path, hdu_list
        except Exception as e:
            raise e
    else:
        hdul = fits.open(file)
        for i, hdu in enumerate(hdul):
            if hdu.data is not None and hdu.is_image:
                hdu_list.append((i, hdu.name if hdu.name else f"HDU {i}"))
        return hdul, 'astropy', None, hdu_list

def extract_image_data(source, index, mode, path=None):
    if mode == 'fz':
        with fitsio.FITS(path) as f:
            data = f[index].read()
            header = dict(f[index].read_header())
    else:
        hdu = source[index]
        data = hdu.data
        header = hdu.header
    return np.nan_to_num(data), header

def generate_png(image_data):
    fig, ax = plt.subplots()
    img = ax.imshow(image_data, cmap='gray', origin='lower')
    ax.set_title("FITS Image")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    plt.colorbar(img, ax=ax, label="Intensity")
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()

def classify_galaxy(data):
    shape = data.shape
    aspect_ratio = shape[0] / shape[1]
    central = np.nanmean(data[int(shape[0]*0.4):int(shape[0]*0.6), int(shape[1]*0.4):int(shape[1]*0.6)])
    outer = np.nanmean(np.concatenate([
        data[:int(shape[0]*0.1), :],
        data[int(shape[0]*0.9):, :],
        data[:, :int(shape[1]*0.1)],
        data[:, int(shape[1]*0.9):]
    ]))
    if aspect_ratio > 1.5 or aspect_ratio < 0.67:
        return "Elliptical (E)", "Elongated shape suggests elliptical galaxy."
    elif central > 2 * outer:
        return "Spiral (S)", "Bright central region suggests spiral structure."
    else:
        return "Irregular (Irr)", "No clear structure detected."

def create_pdf_report(header, classification, description, mean_brightness, std_brightness, image_bytes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Galaxy FITS Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    for k, v in header.items():
        if isinstance(v, (str, int, float)):
            pdf.cell(0, 8, txt=f"{k}: {v}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 8, txt=f"Classification: {classification}", ln=True)
    pdf.cell(0, 8, txt=f"Description: {description}", ln=True)
    pdf.cell(0, 8, txt=f"Mean Brightness: {mean_brightness:.2f}", ln=True)
    pdf.cell(0, 8, txt=f"Std Brightness: {std_brightness:.2f}", ln=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
        tmp_img.write(image_bytes)
        tmp_img_path = tmp_img.name

    pdf.image(tmp_img_path, w=180)
    os.remove(tmp_img_path)

    buf = BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ---- 실행 ----

if uploaded_file:
    st.write("📂 Loading FITS file...")
    try:
        fits_source, mode, temp_path, hdu_list = load_fits_data(uploaded_file, uploaded_file.name)
        hdu_options = [f"{i} - {label}" for i, label in hdu_list]
        hdu_index = st.selectbox("Select Image HDU", options=hdu_options)
        selected_index = int(hdu_index.split(" - ")[0])
        image_data, header = extract_image_data(fits_source, selected_index, mode, temp_path)

        # 이미지 정보
        shape = image_data.shape
        mean = np.mean(image_data)
        std = np.std(image_data)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Image Stats")
            st.write(f"**Size**: {shape}")
            st.write(f"**Mean Brightness**: {mean:.2f}")
            st.write(f"**Std Dev**: {std:.2f}")
            if 'EXPTIME' in header:
                st.write(f"**Exposure**: {header['EXPTIME']} sec")
        with col2:
            st.subheader("🖼️ Image")
            image_png = generate_png(image_data)
            st.image(image_png, caption="Galaxy FITS Image", use_column_width=True)
            st.download_button("📥 Download Image (PNG)", image_png, file_name="galaxy_image.png", mime="image/png")

        # 허블 분류
        st.subheader("🔍 Classification")
        cls, desc = classify_galaxy(image_data)
        st.write(f"**{cls}** – {desc}")

        # 천체 좌표
        st.sidebar.header("📌 Sky Coordinates (Seoul)")
        ra = header.get("RA")
        dec = header.get("DEC")
        if ra and dec:
            try:
                coord = SkyCoord(ra=ra, dec=dec, unit=('hourangle', 'deg'))
                altaz = coord.transform_to(AltAz(obstime=now, location=seoul_location))
                st.sidebar.metric("Altitude (°)", f"{altaz.alt:.2f}")
                st.sidebar.metric("Azimuth (°)", f"{altaz.az:.2f}")
            except Exception as e:
                st.sidebar.warning(f"Coordinate error: {e}")
        else:
            st.sidebar.info("RA/DEC not in header.")

        # 헤더 출력
        st.sidebar.subheader("📄 FITS Header")
        st.sidebar.dataframe(pd.DataFrame(header.items(), columns=["Key", "Value"]))

        # PDF 보고서 생성
        st.subheader("📑 Generate Report")
        if st.button("📄 Generate PDF Report"):
            pdf_data = create_pdf_report(header, cls, desc, mean, std, image_png)
            st.download_button("📥 Download PDF", data=pdf_data, file_name="galaxy_report.pdf", mime="application/pdf")

        # 정리
        if mode == 'fz':
            fits_source.close()
            os.remove(temp_path)

    except Exception as e:
        st.error(f"🚨 Error: {e}")
else:
    st.info("Please upload a FITS file to begin analysis.")
