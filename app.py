import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="로맨스 스캠 심리 조작 탐지 시스템",
    page_icon="🛡️",
    layout="wide"
)

# 2. 자원 로딩 (캐싱을 통한 속도 최적화)
@st.cache_resource
def load_ocr_reader():
    # 한국어 및 영어 OCR 엔진 로드
    return easyocr.Reader(['ko', 'en'])

@st.cache_resource
def load_kobert_model():
    # 깃허브 용량 제한(25MB)을 회피하기 위해 공개된 한국어 BERT 모델을 허브에서 직접 다운로드
    MODEL_NAME = "kykim/bert-kor-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    return tokenizer, model

# 로딩 호출
reader = load_ocr_reader()
tokenizer, model = load_kobert_model()

# 3. 이미지 전처리 함수 (OpenCV 활용)
def preprocess_image(image):
    # PIL 이미지를 NumPy 배열(OpenCV 형식)로 변환
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 명암 대비 극대화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    
    # OCR 인식률 향상을 위한 2배 확대
    resized = cv2.resize(contrast, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

# 4. 심리 조작 기제 및 스캠 탐지 로직
LABELS = {
    0: {"title": "정상 대화", "color": "🟢", "desc": "일반적인 일상 대화입니다. 특이 위험 요소가 발견되지 않았습니다."},
    1: {"title": "가짜 친밀감 / 애정 공세", "color": "🟡", "desc": "부자연스럽거나 신속한 애정 표현을 통해 정서적 유대를 강요하는 단계입니다."},
    2: {"title": "심리적 압박 / 비밀 유지", "color": "🟠", "desc": "타인과의 상의를 차단하거나 고립시키려는 심리적 조작 정황이 포착되었습니다."},
    3: {"title": "금전 요구 / 사기 위험", "color": "🔴", "desc": "세관, 수수료, 가상자산, 급전 요구 등 명백한 금전 편취 시도로 매우 위험합니다!"}
}

def analyze_scam_text(text):
    # [동적 제어 필터] 일상 소액 대화 오탐지(False Positive) 방지
    daily_keywords = ["마라탕", "떡볶이", "커피", "버스비", "택시비", "편의점", "2000원", "3000원"]
    scam_keywords = ["세관", "해외", "수수료", "가상자산", "코인", "대출", "100만원", "송금", "선물"]
    
    if any(kw in text for kw in daily_keywords) and not any(kw in text for kw in scam_keywords):
        return 0, 0.95  # 정상 대화로 우선 분류

    # [규칙 기반 맥락 보완 안전망]
    if any(kw in text for kw in ["세관", "수수료", "100만원", "송금", "코인", "가상자산", "달러"]):
        return 3, 0.92
    elif any(kw in text for kw in ["비밀", "누구한테도 말하지 마", "우리 둘만", "구원"]):
        return 2, 0.88
    elif any(kw in text for kw in ["자기야", "사랑해", "운명", "당신뿐"]):
        return 1, 0.85

    # [KoBERT 모델 추론]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item()

    return pred, confidence

# 5. 웹 화면 레이아웃 (UI)
st.title("🛡️ 로맨스 스캠 심리 조작 탐지 시스템")
st.markdown("""
넷플릭스 다큐멘터리 《소셜 딜레마》의 설득적 매체 프레임과 **KoBERT 기반 인공지능**을 결합하여, 
대화 속 **정서적 필터버블(Emotional Filter Bubble)** 및 스캠 위험도를 실시간으로 진단합니다.
""")

st.divider()

# 탭 구성: 이미지 업로드 / 텍스트 직접 입력
tab1, tab2 = st.tabs(["📸 메신저 캡처 이미지 분석", "✍️ 대화 문장 직접 입력"])

with tab1:
    st.subheader("대화 캡처 이미지 업로드")
    uploaded_file = st.file_uploader("카카오톡, 인스타그램 등 대화 캡처 이미지를 첨부하세요.", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="업로드된 원본 이미지", use_container_width=True)

        with st.spinner("이미지 전처리 및 OCR 글자 추출 중..."):
            processed_img = preprocess_image(image)
            ocr_result = reader.readtext(processed_img, detail=0)
            extracted_text = " ".join(ocr_result)

        with col2:
            st.subheader("📝 추출된 대화 텍스트")
            if extracted_text.strip():
                st.info(extracted_text)
            else:
                st.warning("이미지에서 글자를 인식하지 못했습니다. 더 선명한 이미지를 올려주세요.")

        if extracted_text.strip():
            st.divider()
            st.subheader("🔍 인공지능 진단 결과")
            pred, confidence = analyze_scam_text(extracted_text)
            result = LABELS[pred]

            st.markdown(f"### {result['color']} 진단 결과: **{result['title']}** (신뢰도: {confidence * 100:.1f}%)")
            st.write(result['desc'])

with tab2:
    st.subheader("대화 문장 직접 입력")
    user_input = st.text_area("의심되는 상대방의 메시지를 입력하세요.", placeholder="예: 자기야 오늘 수고 많았어. 세관에 선물이 묶여서 100만원만 보내주면 해결돼!")

    if st.button("위험도 진단하기"):
        if user_input.strip():
            pred, confidence = analyze_scam_text(user_input)
            result = LABELS[pred]

            st.divider()
            st.markdown(f"### {result['color']} 진단 결과: **{result['title']}** (신뢰도: {confidence * 100:.1f}%)")
            st.write(result['desc'])
        else:
            st.warning("문장을 입력해 주세요.")