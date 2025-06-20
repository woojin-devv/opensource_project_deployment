import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models, torchvision.transforms as transforms
import google.generativeai as genai  
import os

# ───────────────────────── 환경 설정 ─────────────────────────
os.environ['PORT'] = '8080'
genai.configure(api_key=st.secrets["gemini_api_key"])

# ───────────────────────── 모델 로딩 ─────────────────────────
@st.cache_resource
def load_yolo_model():
    return YOLO('best.pt')

@st.cache_resource
def load_portion_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state_dict = torch.load('portion_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ───────────────────────── 데이터 로딩 ─────────────────────────
@st.cache_data
def load_nutrition_data():
    df = pd.read_csv("nutrition_db.csv")
    df['음식명_lower'] = df['음식명'].str.strip().str.lower()
    return df

nutrition_df = load_nutrition_data()

# ───────────────────────── 유틸 함수 ─────────────────────────
def preprocess_portion_image(image_np):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image_np).unsqueeze(0)

def calculate_bmi(weight_kg, height_cm):
    h = height_cm / 100
    return round(weight_kg / (h * h), 1)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "저체중"
    elif bmi < 23:
        return "정상"
    elif bmi < 25:
        return "과체중"
    else:
        return "비만"

def estimate_caloric_needs(age, gender, height_cm, weight_kg):
    if gender == "남성":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender == "여성":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return round(bmr * 1.55)

def generate_nutrition_feedback_gemini(bmi, bmi_status, nutrition_df, age, gender, height_cm, weight_kg):
    summary = "\n".join(f"{row['영양소']}: {row['합계']:.2f}" for _, row in nutrition_df.iterrows())

    prompt = f"""
    다음은 한 사용자의 건강 정보 및 섭취 영양소입니다.

    [사용자 정보]
    - 나이: {age}세
    - 성별: {gender}
    - 키: {height_cm} cm
    - 몸무게: {weight_kg} kg
    - BMI: {bmi} ({bmi_status})

    [섭취한 영양소]
    {summary}

    위 정보를 바탕으로 이 사용자에게 맞는 건강한 식단 개선 조언을 한국어로 3~5줄 이내로 제안해주세요.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text



# ───────────────────────── Streamlit UI ─────────────────────────
st.set_page_config(page_title='🍱 식단 이미지 분석기', layout='wide')
st.markdown("""
<h1 style='text-align:center;'>🍱 식단 이미지 분석기</h1>
<p style='text-align:center;'>YOLOv8으로 음식 탐지, ResNet50으로 섭취량 예측, 영양정보 + Gemini 피드백까지!</p>
""", unsafe_allow_html=True)

# ───────────────────────── 사이드바 입력 ─────────────────────────
st.sidebar.header("🤖 사용자 정보 입력")
age = st.sidebar.number_input("나이", min_value=0, max_value=120, value=25)
gender = st.sidebar.selectbox("성별", ["남성", "여성", "기타"])
height = st.sidebar.number_input("키(cm)", min_value=50, max_value=250, value=170)
weight = st.sidebar.number_input("몸무게(kg)", min_value=20, max_value=200, value=60)

bmi = calculate_bmi(weight, height)
bmi_status = get_bmi_category(bmi)
recommended_kcal = estimate_caloric_needs(age, gender, height, weight)

st.sidebar.markdown(f"""
✨ **개인 건강 정보**
- BMI: **{bmi}** ({bmi_status})
- 권장 일일 칼로리: **{recommended_kcal} kcal**
""")

st.sidebar.header("📤 이미지 업로드")
uploaded = st.sidebar.file_uploader("이미지를 업로드하세요", ["jpg","jpeg","png"])

model = load_yolo_model()
portion_model = load_portion_model()

# ───────────────────────── 메인 로직 ─────────────────────────
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    rgb = np.array(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    st.image(rgb, caption="🖼️ 업로드 이미지", use_container_width=True)

    res = model.predict(bgr, conf=0.25, verbose=False)[0]
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    boxes = res.boxes.xyxy.cpu().numpy()
    names = model.names

    st.subheader("🍽️ 인식된 항목")
    detected = {}
    for cid in cls_ids:
        n = names[cid]
        detected[n] = detected.get(n, 0) + 1
    if detected:
        for n, cnt in detected.items():
            st.write(f"- {n}: {cnt}")
    else:
        st.warning("❗ 인식된 음식이 없습니다.")
    st.image(res.plot(), caption="YOLO 탐지 결과", use_container_width=True)

    st.subheader("🔍 객체별 섭취량 + 영양정보")
    total_nutrition = None

    for idx, (cid, box) in enumerate(zip(cls_ids, boxes), start=1):
        x1, y1, x2, y2 = map(int, box)
        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0: continue

        label = names[cid].strip().lower().replace(" ", "")
        m = nutrition_df[nutrition_df['음식명_lower'] == label]
        kor = m['한글명'].values[0] if not m.empty else label

        st.markdown(f"### 🍛 {idx}. {kor}")
        c1, c2 = st.columns([1,2])
        c1.image(crop, width=250)
        with c2:
            try:
                t = preprocess_portion_image(crop)
                with torch.no_grad():
                    val = portion_model(t).item()
                st.success(f"🥄 섭취량 점수: **{val:.2f}**")

                if not m.empty:
                    data = m.iloc[0][1:-2].astype(float) * val
                    total_nutrition = data.copy() if total_nutrition is None else total_nutrition + data
                    df = pd.DataFrame(data).reset_index()
                    df.columns = ["영양소", "예상 섭취량"]
                    st.dataframe(df, use_container_width=True, height=300)
                else:
                    st.info("⚠️ 해당 음식의 영양정보가 없습니다.")
            except Exception as e:
                st.warning(f"분석 오류: {e}")
        st.markdown("---")

    if total_nutrition is not None:
        st.subheader("🧠 Gemini로부터 받은 영양 피드백")
        tdf = total_nutrition.reset_index()
        tdf.columns = ["영양소", "합계"]
        st.dataframe(tdf, use_container_width=True)

        with st.spinner("Gemini에게 요청 중..."):
            fb = generate_nutrition_feedback_gemini(bmi, bmi_status, tdf, age, gender, height, weight)
        st.success("✅ 피드백 완료")
        st.markdown(f"**🍀 맞춤 피드백:**\n\n{fb}")
