import streamlit as st
import numpy as np
import pandas as pd
from astropy.io import fits
from PIL import Image
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from datetime import datetime
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from difflib import get_close_matches
import pydeck as pdk
from astropy.visualization import simple_norm

# --- 페이지 설정 ---
st.set_page_config(page_title="천문 이미지 분석기 (ML 은하 분류)", layout="wide")
st.title("🔭 천문 이미지 처리 및 머신러닝 은하 분류 앱")

# --- 1) 대용량 학습 데이터 로드 및 모델 학습 ---
DATA_URL = (
    "https://raw.githubusercontent.com/astronomy-ml/galaxy-zoo-dataset/"
    "master/galaxy_features.csv"
)

@st.cache_resource
def load_and_train_model(data_url):
    df = pd.read_csv(data_url)
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
    return model

model = load_and_train_model(DATA_URL)

# 은하 라벨 맵핑 (예시, 학습 데이터 라벨 확인 후 필요 시 변경)
label_map = {
    0: "타원은하 (Elliptical)",
    1: "나선은하 (Spiral)",
    2: "불규칙은하 (Irregular)"
}

# --- 2) 관측소 DB & 위치 설정 ---
observatory_db = {
    "KECK": {"name":"Keck Observatory","lat":19.8283,"lon":-155.4781},
    "VLT":  {"name":"Very Large Telescope","lat":-24.6270,"lon":-70.4045},
    "SUBARU":{"name":"Subaru Telescope", "lat":19.825,"lon":-155.4761},
    "KPNO": {"name":"Kitt Peak Nat. Obs.","lat":31.9583,"lon":-111.5983}
}
seoul_location = EarthLocation(lat=37.5665, lon=126.9780, height=50)
now_astropy = Time(datetime.utcnow())

# --- 3) 전처리 및 유틸 함수 ---
def robust_preprocess(data):
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if np.all(data==data.flat[0]):
        raise ValueError("이미지가 균일하여 분석에 부적합합니다.")
    lower,upper = np.percentile(data,[2,98])
    return np.clip(data, lower, upper)

def extract_features(data):
    h,w = data.shape
    mean_b = data.mean()
    median_b = np.median(data)
    std_b = data.std()
    skew_b = skew(data.flatten())
    aspect = h/w
    center = data[h//3:2*h//3, w//3:2*w//3]
    c_mean = center.mean()
    outer = (mean_b*h*w - c_mean*center.size)/(h*w - center.size)
    concentrate = c_mean/(outer+1e-5)
    return [mean_b, median_b, std_b, skew_b, concentrate, aspect]

def match_observatory(tname, db):
    keys = list(db.keys())
    name = tname.upper().strip()
    m = get_close_matches(name, keys, n=1, cutoff=0.6)
    if m: return db[m[0]]
    for k in keys:
        if k in name: return db[k]
    return None

# --- 4) FITS 업로드 & 분석 ---
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
        col1,col2 = st.columns(2)

        with col1:
            st.header("이미지 메타정보")
            st.text(f"크기: {data.shape[1]} x {data.shape[0]} px")
            if 'OBJECT' in hdr: st.text(f"대상: {hdr['OBJECT']}")
            if 'EXPTIME' in hdr: st.text(f"노출: {hdr['EXPTIME']} s")

            st.header("물리량 & 분류")
            st.metric("평균 밝기", f"{data.mean():.2f}")
            feats = extract_features(data)
            pred = model.predict([feats])[0]
            pred_label = label_map.get(pred, str(pred))
            st.metric("예측 은하 유형", pred_label)

            # 예측 확률 표시 (상위 3개)
            proba = model.predict_proba([feats])[0]
            top_idx = np.argsort(proba)[::-1][:3]
            proba_text = "\n".join([f"{label_map.get(i,i)}: {proba[i]*100:.1f}%" for i in top_idx])
            st.text_area("분류 확률", proba_text, height=100)

        with col2:
            st.header("이미지 미리보기")
            norm = simple_norm(data, 'sqrt', percent=99)
            img = (norm(data)*255).astype(np.uint8)
            st.image(Image.fromarray(img), use_container_width=True)

        # 사이드바: RA/DEC → AltAz
        st.sidebar.header("🧭 천체 위치 (서울)")
        if 'RA' in hdr and 'DEC' in hdr:
            try:
                # RA 단위 시도 (hourangle 우선, 실패 시 degree)
                try:
                    coord = SkyCoord(ra=hdr['RA'], dec=hdr['DEC'], unit=('hourangle','deg'))
                except Exception:
                    coord = SkyCoord(ra=hdr['RA'], dec=hdr['DEC'], unit=('deg','deg'))

                azel = coord.transform_to(AltAz(obstime=now_astropy, location=seoul_location))
                st.sidebar.metric("고도 (°)", f"{azel.alt.degree:.2f}")
                st.sidebar.metric("방위각 (°)", f"{azel.az.degree:.2f}")
            except Exception as e:
                st.sidebar.warning(f"위치 계산 실패: {e}")
        else:
            st.sidebar.info("RA/DEC 정보가 없습니다.")

        # 관측소 맵핑
        st.subheader("🗺️ 관측소 위치 표시")
        tel = hdr.get('TELESCOP','').upper()
        obs = match_observatory(tel, observatory_db)
        if obs:
            st.markdown(f"**관측소:** {obs['name']}")
            data_layer = [{'lon': obs['lon'], 'lat': obs['lat']}]
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=obs['lat'], longitude=obs['lon'], zoom=4, pitch=0
                ),
                layers=[pdk.Layer(
                    'ScatterplotLayer',
                    data=data_layer,
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
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.comments.append((name.strip(), comment.strip(), timestamp))
            st.success("댓글이 저장되었습니다.")
        else:
            st.warning("이름과 댓글을 모두 입력해주세요.")

if st.session_state.comments:
    st.subheader("📋 전체 댓글")
    # 댓글 박스에 스크롤 가능하도록 스타일 추가
    st.markdown("""
    <style>
    .comment-box {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 8px;
        border-radius: 5px;
        background-color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)

    comment_md = ""
    for i,(n,c,t) in enumerate(reversed(st.session_state.comments),1):
        comment_md += f"**{i}. {n}** ({t}): {c}\n\n"
    st.markdown(f'<div class="comment-box">{comment_md}</div>', unsafe_allow_html=True)
else:
    st.info("아직 댓글이 없습니다. 첫 댓글을 남겨보세요!")
