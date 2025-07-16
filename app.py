import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from skimage.measure import label, regionprops
import tempfile
import pandas as pd
import seaborn as sns
import io
import base64

try:
    from photutils.detection import DAOStarFinder
    from photutils.aperture import CircularAperture
except ImportError:
    DAOStarFinder = None
    CircularAperture = None

st.set_page_config(page_title="은하 FITS 분석기", layout="wide")
st.title("\U0001F30C 은하 FITS 파일 분석 웹앱")

uploaded_file = st.file_uploader("FITS 또는 FZ 파일 업로드", type=["fits", "fit", "fz"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        try:
            hdul = fits.open(tmp.name)
        except Exception as e:
            st.error(f"FITS 파일 열기 실패: {e}")
            st.stop()

        hdu_img = None
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                hdu_img = hdu
                break

        if hdu_img is None:
            st.error("이미지 데이터를 포함한 HDU를 찾을 수 없습니다.")
            st.stop()

        image_data = hdu_img.data
        header = hdu_img.header

        st.subheader("\U0001F4C4 FITS 기본 정보")
        st.write(f"관측 대상: {header.get('OBJECT', '알 수 없음')}")
        st.write(f"관측일자: {header.get('DATE-OBS', '미기록')}")
        st.write(f"망원경: {header.get('TELESCOP', '미기록')}")
        st.write(f"필터: {header.get('FILTER', '미기록')}")
        st.write(f"노출 시간 (초): {header.get('EXPTIME', '미기록')}")
        st.write(f"이미지 차원: {image_data.shape}")
        st.write(f"스펙트럼 정보 있음: {'예' if image_data.ndim >= 3 else '아니오'}")

        wcs = None
        try:
            wcs = WCS(header)
        except Exception:
            pass

        st.subheader("\U0001F52C 은하 이미지 시각화")
        fig, ax = plt.subplots(figsize=(6, 6))
        norm = simple_norm(image_data, 'sqrt', percent=99)
        ax.imshow(image_data[0] if image_data.ndim == 3 else image_data, cmap='gray', norm=norm, origin='lower')
        ax.set_title("은하 이미지")
        st.pyplot(fig)

        st.subheader("\U0001F52D 은하 구조 분석")
        mean, std = np.mean(image_data), np.std(image_data)
        threshold = mean + 3 * std
        binary = (image_data[0] if image_data.ndim == 3 else image_data) > threshold
        labeled_img = label(binary)
        regions = regionprops(labeled_img)

        if len(regions) > 0:
            regions.sort(key=lambda r: r.area, reverse=True)
            galaxy = regions[0]
            st.write(f"- 면적 (픽셀 수): {galaxy.area}")
            st.write(f"- 추정 반지름: {np.sqrt(galaxy.area / np.pi):.1f} px")
            st.write(f"- 중심 좌표: {galaxy.centroid}")
            st.write(f"- 타원률 (0 = 원형): {galaxy.eccentricity:.2f}")
        else:
            st.write("은하로 추정되는 구조를 찾지 못했습니다.")

        st.subheader("\U0001F30E 거리 및 위치 정보")
        ra, dec = header.get('RA'), header.get('DEC')
        z = header.get('REDSHIFT') or header.get('Z')
        if ra and dec:
            st.write(f"- 적경(RA): {ra}")
            st.write(f"- 적위(DEC): {dec}")
        if z:
            try:
                c = 3e5
                H0 = 70
                distance = (float(z) * c) / H0
                st.write(f"- 허블 거리 추정: {distance:.1f} Mpc")
            except:
                st.write("- 적색편이 값이 숫자가 아닙니다.")

        st.subheader("\U0001F6F0️ 운동 및 활동성 해석")
        if image_data.ndim >= 3:
            velocity_map = image_data[-1] - image_data[0]
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            im = ax2.imshow(velocity_map, cmap='RdBu_r', origin='lower')
            fig2.colorbar(im, ax=ax2, label='상대 속도')
            ax2.set_title("스펙트럼 큐브 속도 분포")
            st.pyplot(fig2)
        else:
            st.write("스펙트럼 큐브가 없어 운동 분석 불가")

        st.subheader("\U0001F52A 화학적 물리적 성질")
        temp = header.get('TEFF') or header.get('TEMP')
        metal = header.get('METAL') or header.get('FE_H')
        sfr = header.get('SFR')
        st.write(f"- 별의 유효 온도: {temp if temp else '정보 없음'}")
        st.write(f"- 금속성 [Fe/H]: {metal if metal else '정보 없음'}")
        st.write(f"- 별 생성률 SFR: {sfr if sfr else '정보 없음'}")

        st.subheader("\U0001F680 활동성 은하핵(AGN) 여부")
        agn = header.get('AGN') or header.get('ACTIVITY')
        if agn:
            st.write(f"- AGN 존재 여부: {agn}")
        else:
            st.write("- AGN 관련 정보 없음")

        st.subheader("\U0001F3A8 H-R 도표 (임의의 색지수 기반)")
        if 'B_MAG' in header and 'V_MAG' in header:
            b = float(header['B_MAG'])
            v = float(header['V_MAG'])
            color_index = b - v
            hr_data = pd.DataFrame({"색지수 B-V": [color_index], "광도": [v]})
            fig_hr, ax_hr = plt.subplots()
            ax_hr.scatter(hr_data["색지수 B-V"], hr_data["광도"], color='blue')
            ax_hr.invert_yaxis()
            ax_hr.set_xlabel("B-V 색지수")
            ax_hr.set_ylabel("밝기 (V)")
            ax_hr.set_title("H-R 도표 위치")
            st.pyplot(fig_hr)

        st.subheader("\U0001F31F 세페이드 변광성 거리 측정 시뮬레이션")
        if 'PERIOD' in header:
            P = float(header['PERIOD'])
            Mv = -2.76 * np.log10(P) - 1.0
            mv = float(header.get('V_MAG', Mv + 10))
            d = 10 ** ((mv - Mv + 5) / 5)
            st.write(f"- 주기: {P} 일")
            st.write(f"- 절대 등급(Mv): {Mv:.2f}")
            st.write(f"- 거리 추정: {d:.2f} pc")

        st.subheader("\U0001FA90 외계행성 트랜싯 시뮬레이션")
        if 'DEPTH' in header and 'DURATION' in header and 'PERIOD' in header:
            depth = float(header['DEPTH'])
            duration = float(header['DURATION'])
            period = float(header['PERIOD'])
            time = np.linspace(0, period, 1000)
            flux = np.ones_like(time)
            center = period / 2
            mask = (time > center - duration/2) & (time < center + duration/2)
            flux[mask] -= depth
            fig_tr, ax_tr = plt.subplots()
            ax_tr.plot(time, flux, color='black')
            ax_tr.set_xlabel("시간 (일)")
            ax_tr.set_ylabel("상대 광도")
            ax_tr.set_title("외계행성 트랜싯 곡선")
            st.pyplot(fig_tr)

        st.subheader("\U0001F4C1 FITS 헤더 전체 보기")
        if st.checkbox("헤더 전체 보기"):
            st.code(str(header))

        st.success("분석 완료! 더 많은 파일을 올려 실험해보세요.")
