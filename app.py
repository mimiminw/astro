import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew
from PIL import Image
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from datetime import datetime
from difflib import get_close_matches
import pydeck as pdk
import os

# --- 0) 가상 데이터 생성 (최초 1회만 실행) ---
os.makedirs("data", exist_ok=True)
csv_path = "data/galaxy_features.csv"
if not os.path.exists(csv_path):
    data = {
        "mean_brightness": [100, 150, 80, 130, 90, 160, 110, 140, 70, 120],
        "median_brightness": [95, 140, 75, 125, 85, 155, 105, 135, 65, 115],
        "std_brightness": [10, 15, 8, 12, 9, 14, 11, 13, 7, 10],
        "skewness": [0.5, 0.3, -0.2, 0.1, 0.4, 0.0, 0.2, 0.3, -0.1, 0.1],
        "concentration": [1.2, 1.5, 0.9, 1.1, 1.0, 1.6, 1.3, 1.4, 0.8, 1.2],
        "aspect_ratio": [1.0, 1.1, 0.9, 1.05, 0.95, 1.15, 1.0, 1.1, 0.85, 1.0],
        "label": [0, 1, 2, 1, 0, 1, 0, 1, 2, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    st.info("가상 데이터셋이 생성되었습니다.")

# --- 1) 모델 학습 함수 ---
@st.cache_resource
def load_and_train_model(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[
        'mean_brightness','median_brightness','std_brightness',
        'skewness','concentration','aspect_ratio','label'
    ])
    X = df[[
        'mean_brightness','median_brightness','std_brightness',
        'skewness','concentration','aspect_ratio'
    ]].values
    y = df['label'].values
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model, df

model, base_df = load_and_train_model(csv_path)

# --- 2) 라벨 맵핑 ---
label_map = {0: "타원은하 (Elliptical)", 1: "나선은하 (Spiral)", 2: "불규칙은하 (Irregular)"}
label_str_to_num = {v:k for k,v in label_map.items()}

# --- 3) 관측소 DB 및 위치 ---
observatory_db = {
    "KECK": {"name":"Keck Observatory","lat":19.8283,"lon":-155.4781},
    "VLT":  {"name":"Very Large Telescope","lat":-24.6270,"lon":-70.4045},
    "SUBARU":{"name":"Subaru Telescope", "lat":19.825,"lon":-155.4761},
    "KPNO": {"name":"Kitt Peak Nat. Obs.","lat":31.9583,"lon":-111.5983}
}
seoul_location = EarthLocation(lat=37.5665, lon=126.9780, height=50)
now_astropy = Time(datetime.utcnow())

# --- 4) 이미지 전처리 및 특징 추출 함수 ---
def robust_preprocess(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(data == data.flat[0]):
        raise ValueError("이미지가 균일하여 분석에 부적합합니다.")
    lower, upper = np.percentile(data, [2, 98])
    return np.clip(data, lower, upper)

def extract_features(data):
    h, w = data.shape
    mean_b = data.mean()
    median_b = np.median(data)
    std_b = data.std()
    skew_b = skew(data.flatten())
    aspect = h / w
    center = data[h//3:2*h//3, w//3:2*w//3]
    c_mean = center.mean()
    outer = (mean_b*h*w - c_mean*center.size) / (h*w - center.size)
    concentrate = c_mean / (outer + 1e-5)
    return [mean_b, median_b, std_b, skew_b, concentrate, aspect]

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

# --- 5) 스트림릿 UI ---
st.set_page_config(page_title="천문 이미지 분석기 (ML 은하 분류)", layout="wide")
st.title("🔭 천문 이미지 처리 및 머신러닝 은하 분류 앱")

# 사용자 데이터 저장용 (세션 상태)
if "user_data" not in st.session_state:
    st.session_state.user_data = []

uploaded = st.file_uploader("분석할 FITS 파일을 선택하세요.", type=['fits','fit','fz'])

if uploaded:
    try:
        hdul = fits.open(uploaded)
        img_hdu = next((h for h in hdul if h.data is not None and h.is_image), None)
        if img_hdu is None:
            st.error("유효한 이미지 HDU를 찾을 수 없습니다.")
            st.stop()

        raw = img_hdu.data
        try:
            data = robust_preprocess(raw)
        except ValueError as e:
            st.warning(str(e))
            st.stop()

        hdr = img_hdu.header
        st.success(f"'{uploaded.name}' 처리 완료")

        col1, col2 = st.columns(2)

        with col1:
            st.header("이미지 메타정보")
            st.text(f"크기: {data.shape[1]} x {data.shape[0]} px")
            if 'OBJECT' in hdr:
                st.text(f"대상: {hdr['OBJECT']}")
            if 'EXPTIME' in hdr:
                st.text(f"노출: {hdr['EXPTIME']} s")

            st.header("물리량 & 분류")
            st.metric("평균 밝기", f"{data.mean():.2f}")
            feats = extract_features(data)
            pred_num = model.predict([feats])[0]
            pred_label = label_map.get(pred_num, str(pred_num))
            st.metric("예측 은하 유형", pred_label)

            # 사용자 라벨 입력
            st.subheader("▶ 실제 은하 유형을 선택하고 학습 데이터에 추가")
            user_label = st.selectbox("실제 은하 유형을 선택하세요", list(label_str_to_num.keys()))
            if st.button("이 데이터로 학습 데이터 추가"):
                st.session_state.user_data.append((feats, label_str_to_num[user_label]))
                st.success(f"데이터가 추가되었습니다! 현재 학습 데이터 수: {len(st.session_state.user_data)}")

            if st.button("모델 재학습"):
                if len(st.session_state.user_data) > 0:
                    # 기존 데이터 + 사용자 데이터 합치기
                    user_feats = np.array([x[0] for x in st.session_state.user_data])
                    user_labels = np.array([x[1] for x in st.session_state.user_data])
                    X = np.vstack([base_df[[
                        'mean_brightness','median_brightness','std_brightness',
                        'skewness','concentration','aspect_ratio'
                    ]].values, user_feats])
                    y = np.concatenate([base_df['label'].values, user_labels])
                    # 모델 재학습
                    model.fit(X, y)
                    st.success("모델이 재학습되었습니다!")
                else:
                    st.warning("추가된 학습 데이터가 없습니다.")

        with col2:
            st.header("이미지 미리보기")
            vmin, vmax = np.percentile(data, [5, 99.5])
            norm = ((np.clip(data, vmin, vmax) - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            st.image(Image.fromarray(norm), use_container_width=True)

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

# --- 6) 댓글 기능 ---
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
