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

# ───────────────────── 설정 ─────────────────────
os.environ['PORT'] = '8080'
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# ───────────────────── 모델 로딩 ─────────────────────
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

# ───────────────────── 데이터 로딩 ─────────────────────
@st.cache_data
def load_nutrition_data():
    df = pd.read_csv("nutrition_db.csv")
    df['음식명_lower'] = df['음식명'].str.strip().str.lower()
    return df

nutrition_df = load_nutrition_data()
quantity_df = pd.read_csv("quantity_level.csv")  # ← Q1~Q5, 비율 0.25~1.25 포함된 csv

# ───────────────────── 유틸 ─────────────────────
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
    if bmi < 18.5: return "저체중"
    elif bmi < 23: return "정상"
    elif bmi < 25: return "과체중"
    else: return "비만"

def estimate_caloric_needs(age, gender, height_cm, weight_kg):
    if gender == "남성":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender == "여성":
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return round(bmr * 1.55)

def calculate_recommended_nutrients(gender, height_cm, activity_level):
    # 1. 적정 체중 계산
    std_weight = height_cm ** 2 * (21 if gender == "여성" else 22) / 10000

    # 2. 권장 에너지 계산
    activity_map = {"활동X": 25, "보통": 30, "활동多": 35}
    kcal = std_weight * activity_map[activity_level]

    # 3. 탄단지 계산 (4:4:2 비율에 해당하는 가중치 적용)
    carb_g = kcal * 0.1
    protein_g = kcal * 0.1
    fat_g = kcal * 0.022

    # 4. 당류: 전체의 10%, 단 최대 50g
    sugar_g = min(kcal * 0.1, 50)

    # 5. 1끼 기준으로 나누기
    return {
        "1일 권장 열량 (kcal)": round(kcal),
        "탄수화물 (g)": round(carb_g),
        "단백질 (g)": round(protein_g),
        "지방 (g)": round(fat_g, 1),
        "당류 최대 (g)": round(sugar_g),
        "1끼 기준 열량 (kcal)": round(kcal / 3),
        "1끼 기준 당류 (g)": round(sugar_g / 3, 1)
    }


def generate_nutrition_feedback_gemini(bmi, bmi_status, nutrition_df, age, gender, height_cm, weight_kg):
    prompt = f"""
    당신은 건강한 식단 개선을 위한 전문가입니다.

    다음은 한 사용자의 건강 정보 및 섭취 영양소입니다.
    [사용자 정보]
    - 나이: {age}세
    - 성별: {gender}
    - 키: {height_cm} cm
    - 몸무게: {weight_kg} kg
    - BMI: {bmi} ({bmi_status})
    - 섭취한 영양소:{nutrition_df.to_string(index=False)}
    

    위 정보를 바탕으로 이 사용자에게 맞는 건강한 식단 개선 조언을 한국어로 3~5줄 이내로 제안해주세요.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# ───────────────────── Streamlit UI ─────────────────────
st.set_page_config(page_title='🍱 식단 이미지 분석기', layout='wide')
st.markdown("<h1 style='text-align:center;'>🍱 식단 이미지 분석기</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; font-size: 16px; color: gray; margin-bottom: 30px;'>
📷 이미지를 업로드하면 AI가 음식 종류를 인식하고, <br>
🥗 섭취량과 영양소를 분석한 뒤, <br>
🧠 Gemini가 맞춤 식단 피드백까지 제공해드립니다.
</div>
""", unsafe_allow_html=True)

# 사이드바
st.sidebar.header("👤 사용자 정보 입력")
age = st.sidebar.number_input("나이", 0, 120, 25)
gender = st.sidebar.selectbox("성별", ["남성", "여성", "기타"])
height = st.sidebar.number_input("키(cm)", 50, 250, 170)
weight = st.sidebar.number_input("몸무게(kg)", 20, 200, 60)
activity = st.sidebar.selectbox("활동 수준", ["활동X", "보통", "활동多"])
st.sidebar.markdown("---")

# 사용자 입력에 따라 동적 계산 표시
st.sidebar.markdown("## 자동 계산된 권장량")


# sidebar 추천량 계산 - 성별, 키, 활동 수준이 모두 입력된 경우에만
if height and gender:
    rec = calculate_recommended_nutrients(gender, height, activity)

    # 에너지 정보는 따로
    st.sidebar.markdown(f"#### 💭 에너지 요약")
    st.sidebar.write(f"- 1일 권장 열량: **{rec['1일 권장 열량 (kcal)']} kcal**")
    st.sidebar.write(f"- 1끼 기준 열량: **{rec['1끼 기준 열량 (kcal)']} kcal**")
    st.sidebar.markdown("---")

    # 권장 영양소 표 만들기 전에 소수점 자리수 조정
    nutrient_table = {
        "탄수화물 (g)": f"{rec['탄수화물 (g)']:.1f}",
        "단백질 (g)": f"{rec['단백질 (g)']:.1f}",
        "지방 (g)": f"{rec['지방 (g)']:.1f}",
        "당류 최대 (g)": f"{rec['당류 최대 (g)']:.1f}",
        "1끼 기준 당류 (g)": f"{rec['1끼 기준 당류 (g)']:.1f}"
    }



    df_nutrients = pd.DataFrame.from_dict(nutrient_table, orient='index', columns=["권장량"])
    st.sidebar.markdown("#### 💭 권장 영양소 (표)")
    st.sidebar.table(df_nutrients)


bmi = calculate_bmi(weight, height)
bmi_status = get_bmi_category(bmi)
recommended_kcal = estimate_caloric_needs(age, gender, height, weight)

st.sidebar.markdown(f"**💭 BMI:** {bmi} ({bmi_status})  \n**💭 일일 권장 칼로리:** {recommended_kcal} kcal")
st.sidebar.markdown("---")
st.sidebar.markdown("## 🍽️ 이미지 업로드")
uploaded = st.sidebar.file_uploader("식단 이미지를 업로드 해주세요.", ["jpg", "jpeg", "png"])

model = load_yolo_model()
portion_model = load_portion_model()

# ───────────────────── 메인 로직 ─────────────────────
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
        name = names[cid]
        detected[name] = detected.get(name, 0) + 1
    if detected:
        for name, count in detected.items():
            st.write(f"- {name}: {count}")
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
        m = nutrition_df[nutrition_df["음식명_lower"] == label]
        kor = m['한글명'].values[0] if not m.empty else label

        st.markdown(f"### 🍛 {idx}. {kor}")
        col1, col2 = st.columns([1, 2])
        col1.image(crop, width=250)

        with col2:
            try:
                input_tensor = preprocess_portion_image(crop)
                with torch.no_grad():
                    score = portion_model(input_tensor).item()
                q_idx = int(round(score))
                q_idx = min(max(q_idx, 0), 4)
                q_label = f"Q{q_idx + 1}"

                ratio_row = quantity_df[quantity_df["알고리즘 결과"] == q_label]
                if ratio_row.empty:
                    st.warning("⚠️ 비율 정보 없음")
                    continue
                ratio = float(ratio_row["양 분석 결과"].values[0])

                st.success(f"🥄 섭취량 등급: **{q_label} (일반식의 {int(ratio * 100)}%)**")

                if not m.empty:
                    base = m.iloc[0]
                    nutrients = base[1:-2].astype(float) * ratio
                    total_nutrition = nutrients if total_nutrition is None else total_nutrition + nutrients
                    df = pd.DataFrame(nutrients).reset_index()
                    df.columns = ["영양소", "예상 섭취량"]
                    st.dataframe(df, use_container_width=True, height=300)
                else:
                    st.info("⚠️ 등록된 영양정보가 없습니다.")
            except Exception as e:
                st.warning(f"분석 오류: {e}")
        st.markdown("---")

    if total_nutrition is not None:
        st.subheader("🧠 Gemini 피드백")
        with st.spinner("Gemini에게 분석 요청 중..."):
            df_total = total_nutrition.reset_index()
            df_total.columns = ["영양소", "합계"]
            feedback = generate_nutrition_feedback_gemini(bmi, bmi_status, df_total, age, gender, height, weight)
        st.success("✅ 피드백 완료")
        st.markdown(f"**🍀 맞춤 피드백:**\n\n{feedback}")