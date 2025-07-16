import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture
from skimage.measure import label, regionprops
from astropy.visualization import simple_norm

st.set_page_config(page_title="Galaxy FITS Analyzer", layout="wide")

st.title("Galaxy FITS File Analyzer")

uploaded_file = st.file_uploader("Upload your Galaxy FITS file", type=["fits", "fit"], "fz")

if uploaded_file is not None:
    # Load FITS file
    hdul = fits.open(uploaded_file)
    
    # Find primary HDU with image data
    hdu_img = None
    for hdu in hdul:
        if hdu.data is not None and hdu.data.ndim >= 2:
            hdu_img = hdu
            break

    if hdu_img is None:
        st.error("No image data found in FITS file.")
        st.stop()
    
    image_data = hdu_img.data
    header = hdu_img.header

    # Show basic info
    st.header("Basic FITS Header Information")
    st.write(f"Object: {header.get('OBJECT', 'Unknown')}")
    st.write(f"Observation Date: {header.get('DATE-OBS', 'Unknown')}")
    st.write(f"Telescope: {header.get('TELESCOP', 'Unknown')}")
    st.write(f"Instrument: {header.get('INSTRUME', 'Unknown')}")
    st.write(f"Filter: {header.get('FILTER', 'Unknown')}")
    st.write(f"Exposure Time (s): {header.get('EXPTIME', 'Unknown')}")

    # Show image with WCS if available
    wcs = None
    try:
        wcs = WCS(header)
    except Exception as e:
        st.warning("WCS info not available or invalid.")

    st.header("Galaxy Image")
    fig, ax = plt.subplots(figsize=(7,7))
    norm = simple_norm(image_data, 'sqrt', percent=99)
    if wcs:
        ax = plt.subplot(projection=wcs)
        ax.imshow(image_data, cmap='gray', norm=norm, origin='lower')
        ax.set_xlabel("RA")
        ax.set_ylabel("DEC")
    else:
        ax.imshow(image_data, cmap='gray', norm=norm, origin='lower')
        ax.set_xlabel("X Pixel")
        ax.set_ylabel("Y Pixel")
    ax.set_title("Galaxy Image")
    st.pyplot(fig)

    # === 1. 은하의 물리적 특성: 크기, 형태, 밝기, 별 구성 ===
    st.header("Galaxy Physical Properties")

    # 간단한 threshold를 통한 은하 영역 분리
    mean, std = np.mean(image_data), np.std(image_data)
    threshold = mean + 3*std
    binary = image_data > threshold

    labeled_img = label(binary)
    regions = regionprops(labeled_img)

    if len(regions) == 0:
        st.write("은하 영역을 찾지 못했습니다.")
    else:
        # 가장 큰 영역을 은하로 가정
        regions.sort(key=lambda r: r.area, reverse=True)
        galaxy_region = regions[0]

        # 크기: 면적, 반경 추정
        area_pix = galaxy_region.area
        radius_pix = np.sqrt(area_pix / np.pi)

        st.write(f"- 은하 픽셀 면적: {area_pix}")
        st.write(f"- 은하 추정 반경 (픽셀 단위): {radius_pix:.2f}")

        # 밝기: 총 밝기, 평균 밝기
        galaxy_mask = labeled_img == galaxy_region.label
        total_flux = np.sum(image_data[galaxy_mask])
        mean_flux = np.mean(image_data[galaxy_mask])

        st.write(f"- 은하 총 밝기 (합산 ADU): {total_flux:.2e}")
        st.write(f"- 은하 평균 밝기: {mean_flux:.2e}")

        # 형태 (타원률) 계산
        eccentricity = galaxy_region.eccentricity
        st.write(f"- 은하 타원률 (0=원형, 1=선형): {eccentricity:.3f}")

    # === 2. 은하의 거리와 위치 (WCS + 적색편이) ===
    st.header("Galaxy Distance & Position")

    ra = header.get('RA')
    dec = header.get('DEC')
    redshift = header.get('REDSHIFT') or header.get('Z') or header.get('Z_V')

    if ra and dec:
        st.write(f"- RA: {ra}")
        st.write(f"- DEC: {dec}")
    else:
        st.write("- RA/DEC 정보가 헤더에 없습니다.")

    if redshift:
        st.write(f"- 적색편이 (Redshift): {redshift}")
        # 거리 계산 예시 (단순 허블법칙)
        H0 = 70  # 허블상수 km/s/Mpc
        c = 3e5  # 광속 km/s
        try:
            z = float(redshift)
            distance_mpc = (c * z) / H0
            st.write(f"- 추정 거리: {distance_mpc:.2f} Mpc (단순 허블법칙 적용)")
        except Exception:
            st.write("- 적색편이 값 변환 실패")
    else:
        st.write("- 적색편이 정보가 헤더에 없습니다.")

    # === 3. 은하 운동 상태 및 상호작용 ===
    st.header("Galaxy Kinematics & Interaction")

    # 간단한 스펙트럼 큐브 데이터가 있다면 속도장 시각화 (예: 3D 데이터)
    if image_data.ndim >= 3:
        st.write("- 스펙트럼 큐브 데이터 발견, 운동 상태 시각화")
        # 예시: 첫번째 파장대 이미지와 마지막 파장대 이미지 차이로 속도장 유추
        velocity_map = image_data[-1] - image_data[0]

        fig2, ax2 = plt.subplots(figsize=(6,6))
        im = ax2.imshow(velocity_map, cmap='RdBu_r', origin='lower')
        ax2.set_title("Estimated Velocity Map (Last - First Slice)")
        fig2.colorbar(im, ax=ax2, label='Velocity (arbitrary units)')
        st.pyplot(fig2)
    else:
        st.write("- 스펙트럼 큐브 데이터가 없어 운동 상태 분석 불가")

    # === 4. 은하 내 별과 가스의 화학적·물리적 상태 ===
    st.header("Chemical & Physical Properties")

    # 스펙트럼 데이터에서 금속성, 온도, 화학적 성분 추정(간단 예시)
    # 실제로는 분광선 분석 필요하므로, 헤더 정보에서 관련 메타데이터 출력

    metalicity = header.get('METALLIC')
    temperature = header.get('TEFF') or header.get('TEMPERAT')

    if metalicity:
        st.write(f"- 금속성(Metallicity): {metalicity}")
    else:
        st.write("- 금속성 정보가 없습니다.")

    if temperature:
        st.write(f"- 별의 유효온도(Effective Temperature): {temperature}")
    else:
        st.write("- 온도 정보가 없습니다.")

    # === 5. 은하 진화 및 활동성 ===
    st.header("Galaxy Evolution & Activity Indicators")

    # AGN(활동성 은하핵) 여부, 별 형성 영역에 대한 메타데이터가 있으면 표시
    agn_flag = header.get('AGN') or header.get('ACTIVITY')
    sf_rate = header.get('SFR')  # Star Formation Rate

    if agn_flag:
        st.write(f"- AGN 활동성 지표: {agn_flag}")
    else:
        st.write("- AGN 관련 정보 없음")

    if sf_rate:
        st.write(f"- 별 형성률(Star Formation Rate): {sf_rate}")
    else:
        st.write("- 별 형성률 정보 없음")

    st.write("**참고:** 상세 분석은 스펙트럼 라인 피팅, 모델링 등을 통해 진행해야 합니다.")

    hdul.close()
