import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from datetime import datetime
from difflib import get_close_matches
import pydeck as pdk
from scipy.ndimage import zoom
from PIL import Image

# --- 설정 ---
st.set_page_config(page_title="CNN 은하 분류 앱 (통합)", layout="wide")
st.title("🔭 CNN 기반 FITS 은하 이미지 분류 및 관측소 분석 앱")

# --- 관측소 DB ---
observatory_db = {
    "KECK": {"name":"Keck Observatory","lat":19.8283,"lon":-155.4781},
    "VLT":  {"name":"Very Large Telescope","lat":-24.6270,"lon":-70.4045},
    "SUBARU":{"name":"Subaru Telescope", "lat":19.825,"lon":-155.4761},
    "KPNO": {"name":"Kitt Peak Nat. Obs.","lat":31.9583,"lon":-111.5983}
}
seoul_location = EarthLocation(lat=37.5665, lon=126.9780, height=50)
now_astropy = Time(datetime.utcnow())

IMG_SIZE = 64

# --- CNN 모델 정의 ---
def create_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- 가상 데이터 생성 ---
def generate_fake_dataset(num_samples=300):
    X = []
    y = []
    for i in range(num_samples):
        label = i % 3
        if label == 0:
            img = np.random.normal(loc=0.7, scale=0.1, size=(IMG_SIZE, IMG_SIZE))
            img = img * np.exp(-((np.indices((IMG_SIZE, IMG_SIZE))[0]-IMG_SIZE//2)**2 + (np.indices((IMG_SIZE, IMG_SIZE))[1]-IMG_SIZE//2)**2)/1000)
        elif label == 1:
            img = np.zeros((IMG_SIZE, IMG_SIZE))
            for angle in range(0, 360, 30):
                x = np.arange(IMG_SIZE)
                y_ = ((np.sin(np.radians(x*5 + angle)) + 1) / 2) * IMG_SIZE
                y_ = y_.astype(int)
                img[(y_ % IMG_SIZE, x % IMG_SIZE)] += 0.7
            img += np.random.normal(scale=0.1, size=(IMG_SIZE, IMG_SIZE))
            img = np.clip(img, 0, 1)
        else:
            img = np.random.rand(IMG_SIZE, IMG_SIZE) * 0.8
            img += np.random.normal(scale=0.3, size=(IMG_SIZE, IMG_SIZE))
            img = np.clip(img, 0, 1)
        X.append(img)
        y.append(label)
    X = np.array(X)[..., np.newaxis].astype(np.float32)
    y = np.array(y)
    return X, y

# --- FITS 전처리 ---
def preprocess_fits_image(hdul):
    img_hdu = next((h for h in hdul if h.data is not None and h.is_image), None)
    if img_hdu is None:
        raise ValueError("유효한 이미지 HDU를 찾을 수 없습니다.")
    data = img_hdu.data
    data = np.nan_to_num(data, nan=0.0)
    lower, upper = np.percentile(data, [5, 99])
    data = np.clip(data, lower, upper)
    data = (data - lower) / (upper - lower)
    zoom_factors = (IMG_SIZE / data.shape[0], IMG_SIZE / data.shape[1])
    data_resized = zoom(data, zoom_factors)
    return data_resized.astype(np.float32)[..., np.newaxis]

# --- 관측소 매칭 ---
def match_observatory(tname, db):
    keys = list(db.keys())
    name = tname.upper().strip()
    m = get_close_matches(name, keys, n=1, cutoff=0.6)
    if m:
        return db[m[0]]
    for k in keys:
        if k in name:
            return db[k]
    return None

# --- 모델 학습 (캐시) ---
@st.cache_resource
def train_model():
    model = create_cnn_model()
    X_train, y_train = generate_fake_dataset()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

model = train_model()
label_names = ["타원은하 (Elliptical)", "나선은하 (Spiral)", "불규칙은하 (Irregular)"]

# --- FITS 업로드 및 처리 ---
uploaded = st.file_uploader("FITS 파일을 업로드하세요", type=['fits','fit','fz'])

if uploaded:
    try:
        hdul = fits.open(uploaded)
        img = preprocess_fits_image(hdul)
        st.image(img[:,:,0], caption="전처리된 이미지 (64x64)", use_container_width=True)

        pred_probs = model.predict(np.expand_dims(img, axis=0))
        pred_class = np.argmax(pred_probs)
        st.metric("예측 은하 유형", label_names[pred_class])
        st.write(f"확률: {pred_probs[0][pred_class]:.3f}")

        hdr = hdul[0].header
        st.header("이미지 메타정보")
        st.text(f"원본 크기: {hdul[0].data.shape[1]} x {hdul[0].data.shape[0]} px")
        if 'OBJECT' in hdr:
            st.text(f"대상: {hdr['OBJECT']}")
        if 'EXPTIME' in hdr:
            st.text(f"노출: {hdr['EXPTIME']} s")

        # 사이드바: RA/DEC → AltAz 변환
        st.sidebar.header("🧭 천체 위치 (서울 기준)")
        if 'RA' in hdr and 'DEC' in hdr:
            try:
                coord = SkyCoord(ra=hdr['RA'], dec=hdr['DEC'], unit=('hourangle','deg'))
                altaz = coord.transform_to(AltAz(obstime=now_astropy, location=seoul_location))
                st.sidebar.metric("고도 (°)", f"{altaz.alt.degree:.2f}")
                st.sidebar.metric("방위각 (°)", f"{altaz.az.degree:.2f}")
            except Exception as e:
                st.sidebar.warning(f"위치 계산 실패: {e}")
        else:
            st.sidebar.info("RA/DEC 정보가 없습니다.")

        # 관측소 위치 표시
        st.subheader("🗺️ 관측소 위치 표시")
        tel = hdr.get('TELESCOP', '').upper()
        obs = match_observatory(tel, observatory_db)
        if obs:
            st.markdown(f"**관측소:** {obs['name']}")
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=obs['lat'], longitude=obs['lon'], zoom=4, pitch=0
                ),
                layers=[pdk.Layer(
                    'ScatterplotLayer',
                    data=[obs],
                    get_position='[lon, lat]',
                    get_color='[200,30,0,160]',
                    get_radius=100000
                )]
            ))
        else:
            st.info("관측소 정보를 찾을 수 없습니다.")
    except Exception as e:
        st.error(f"파일 처리 중 오류: {e}")
else:
    st.info("시작하려면 FITS 파일을 업로드하세요.")

# --- 댓글 기능 ---
st.divider()
st.header("💬 의견 남기기")
if "comments" not in st.session_state:
    st.session_state.comments = []
with st.form("comment_form"):
    name = st.text_input("이름", key="name_input")
    comment = st.text_area("댓글", key="comment_input")
    if st.form_submit_button("댓글 남기기"):
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
