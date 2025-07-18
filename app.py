import streamlit as st
import numpy as np
from astropy.io import fits
from PIL import Image
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from datetime import datetime
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
import pydeck as pdk

# --- 페이지 설정 ---
st.set_page_config(page_title="천문 이미지 분석기 (ML 은하 분류)", layout="wide")
st.title("🔭 천문 이미지 처리 및 머신러닝 은하 분류 앱")

# --- 서울 위치 및 현재 시간 ---
seoul_location = EarthLocation(lat=37.5665, lon=126.9780, height=50)
now = datetime.utcnow()
now_astropy = Time(now)

# --- 관측소 DB ---
observatory_db = {
    "KECK": {"name": "Keck Observatory", "lat": 19.8283, "lon": -155.4781},
    "VLT": {"name": "Very Large Telescope", "lat": -24.6270, "lon": -70.4045},
    "SUBARU": {"name": "Subaru Telescope", "lat": 19.825, "lon": -155.4761},
    "KPNO": {"name": "Kitt Peak National Observatory", "lat": 31.9583, "lon": -111.5983}
}

# --- 머신러닝 모델 (가상 학습 데이터) ---
def train_sample_model():
    X_train = [
        [6000, 5800, 300, 0.1, 2.0, 1.0],  # 타원은하
        [3500, 3300, 600, 1.0, 1.0, 1.5],  # 나선은하
        [1000, 900, 900, 0.5, 0.8, 1.2],   # 불규칙은하
        [7000, 6800, 250, 0.2, 2.2, 1.1],  # 타원은하
        [3000, 2800, 700, 1.2, 0.9, 1.6],  # 나선은하
        [900, 800, 850, 0.4, 0.7, 1.0],    # 불규칙은하
    ]
    y_train = [
        "타원은하 (E형)",
        "나선은하 (S형)",
        "불규칙은하 (Irr형)",
        "타원은하 (E형)",
        "나선은하 (S형)",
        "불규칙은하 (Irr형)"
    ]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_sample_model()

# --- 특징 추출 함수 ---
def extract_features(data):
    height, width = data.shape
    mean_brightness = np.mean(data)
    median_brightness = np.median(data)
    std_brightness = np.std(data)
    skewness = skew(data.flatten())
    aspect_ratio = height / width

    center_slice = data[height//3:2*height//3, width//3:2*width//3]
    center_mean = np.mean(center_slice)
    outer_mean = (np.mean(data)*height*width - center_mean*center_slice.size) / (height*width - center_slice.size)
    concentration = center_mean / (outer_mean + 1e-5)

    return [mean_brightness, median_brightness, std_brightness, skewness, concentration, aspect_ratio]

# --- 파일 업로드 ---
uploaded_file = st.file_uploader("분석할 FITS 파일을 선택하세요.", type=['fits', 'fit', 'fz'])

if uploaded_file:
    try:
        with fits.open(uploaded_file) as hdul:
            image_hdu = None
            for hdu in hdul:
                if hdu.data is not None and hdu.is_image:
                    image_hdu = hdu
                    break

            if image_hdu is None:
                st.error("파일에서 유효한 이미지 데이터를 찾을 수 없습니다.")
            else:
                header = image_hdu.header
                data = image_hdu.data
                data = np.nan_to_num(data)

                st.success(f"**'{uploaded_file.name}'** 파일을 성공적으로 처리했습니다.")

                col1, col2 = st.columns(2)

                with col1:
                    st.header("이미지 정보")
                    st.text(f"크기: {data.shape[1]} x {data.shape[0]} 픽셀")
                    if 'OBJECT' in header:
                        st.text(f"관측 대상: {header['OBJECT']}")
                    if 'EXPTIME' in header:
                        st.text(f"노출 시간: {header['EXPTIME']} 초")

                    st.header("물리량")
                    mean_brightness = np.mean(data)
                    st.metric(label="이미지 전체 평균 밝기", value=f"{mean_brightness:.2f}")

                    # 머신러닝 은하 분류
                    features = extract_features(data)
                    classification = model.predict([features])[0]
                    st.metric(label="머신러닝 예측 은하 유형", value=classification)

                with col2:
                    st.header("이미지 미리보기")
                    if data.max() == data.min():
                        norm_data = np.zeros(data.shape, dtype=np.uint8)
                    else:
                        scale_min = np.percentile(data, 5)
                        scale_max = np.percentile(data, 99.5)
                        data_clipped = np.clip(data, scale_min, scale_max)
                        norm_data = (255 * (data_clipped - scale_min) / (scale_max - scale_min)).astype(np.uint8)

                    img = Image.fromarray(norm_data)
                    st.image(img, caption="업로드된 FITS 이미지", use_container_width=True)

                # 사이드바: 천체 위치 계산 (서울 기준)
                st.sidebar.header("🧭 현재 천체 위치 (서울 기준)")
                if 'RA' in header and 'DEC' in header:
                    try:
                        target_coord = SkyCoord(ra=header['RA'], dec=header['DEC'], unit=('hourangle', 'deg'))
                        altaz = target_coord.transform_to(AltAz(obstime=now_astropy, location=seoul_location))
                        st.sidebar.metric("고도 (°)", f"{altaz.alt.degree:.2f}")
                        st.sidebar.metric("방위각 (°)", f"{altaz.az.degree:.2f}")
                    except Exception as e:
                        st.sidebar.warning(f"천체 위치 계산 실패: {e}")
                else:
                    st.sidebar.info("FITS 헤더에 RA/DEC 정보가 없습니다.")

                # 관측소 위치 시각화
                st.subheader("🗺️ 관측소 위치 표시")
                tele_name = header.get('TELESCOP', '').upper().strip()
                st.write(f"TELESCOP 헤더 값: '{tele_name}'")

                observatory_found = None
                for key in observatory_db:
                    if key in tele_name:
                        observatory_found = observatory_db[key]
                        break

                if observatory_found:
                    st.markdown(f"**관측소 이름:** {observatory_found['name']}")
                    st.pydeck_chart(pdk.Deck(
                        initial_view_state=pdk.ViewState(
                            latitude=observatory_found['lat'],
                            longitude=observatory_found['lon'],
                            zoom=4,
                            pitch=0,
                        ),
                        layers=[
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=[observatory_found],
                                get_position='[lon, lat]',
                                get_color='[200, 30, 0, 160]',
                                get_radius=100000,
                            )
                        ]
                    ))
                else:
                    st.info("관측소 정보를 찾을 수 없거나, 헤더에 'TELESCOP' 정보가 없습니다.")

    except Exception as e:
        st.error(f"파일 처리 중 오류가 발생했습니다: {e}")
        st.warning("파일이 손상되었거나 유효한 FITS 형식이 아닐 수 있습니다.")
else:
    st.info("시작하려면 FITS 파일을 업로드해주세요.")

# --- 댓글 기능 ---
st.divider()
st.header("💬 의견 남기기")
if "comments" not in st.session_state:
    st.session_state.comments = []

with st.form(key="comment_form"):
    name = st.text_input("이름을 입력하세요", key="name_input")
    comment = st.text_area("댓글을 입력하세요", key="comment_input")
    submitted = st.form_submit_button("댓글 남기기")
    if submitted:
        if name.strip() and comment.strip():
            st.session_state.comments.append((name.strip(), comment.strip()))
            st.success("댓글이 저장되었습니다.")
        else:
            st.warning("이름과 댓글을 모두 입력해주세요.")

if st.session_state.comments:
    st.subheader("📋 전체 댓글")
    for i, (n, c) in enumerate(reversed(st.session_state.comments), 1):
        st.markdown(f"**{i}. {n}**: {c}")
else:
    st.info("아직 댓글이 없습니다. 첫 댓글을 남겨보세요!")
