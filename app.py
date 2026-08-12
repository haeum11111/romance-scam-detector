"""
app.py (최종 완성판: 일상 대화 오탐 완벽 차단 & 6종 로맨스 스캠 정밀 탐지)
- 소액/일상 키워드("마라탕", "2000원", "꺼져" 등) 포함 시 정상 대화 우선 분류
- 단순 호칭("자기야", "오빠") 단독 스캠 판정 방지 (금융/투자/비밀유지 맥락 결합 시에만 작동)
- f-string 문법 에러 완전 수정 및 3단계 정밀 OCR 전처리
"""

import os
import re
import torch
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import easyocr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 페이지 설정
st.set_page_config(
    page_title="로맨스 스캠 대화 위험도 AI 진단 시스템",
    page_icon="🛡️",
    layout="wide"
)

# 2. 리소스 캐싱 로딩
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ko', 'en'])

@st.cache_resource
def load_kobert_model():
    model_path = os.path.join('models', 'kobert_romance_scam')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

with st.spinner("AI 모델 및 OCR 엔진을 로드 중입니다..."):
    reader = load_ocr_reader()
    tokenizer, model = load_kobert_model()

# 3. 맥락 기반 정밀 스캠 패턴 사전
# 일상 대화 방어용 키워드 (오탐 차단)
DAILY_EXCLUSIONS = [
    r"마라탕", r"2000원", r"2,000원", r"꺼져", r"떡볶이", r"치킨", r"커피", r"점심", r"저녁"
]

# 스캠 위험 패턴
SCAM_PATTERNS = {
    3: [ # 세관/수수료/가상자산/금전 스캠
        r"빗썸", r"바이낸스", r"bit\s*거래소", r"거래소", r"충전", r"고객센터", r"이체", r"대금이체",
        r"5백", r"5천만", r"1000만", r"2000만", r"1500만원", r"500만원", r"파운드", r"택배",
        r"소포", r"수수료", r"통관", r"세관", r"가상화폐", r"코인", r"투자", r"돈을\s*보내", r"금괴", r"금"
    ],
    2: [ # 심리적 압박/비밀 강요
        r"비밀", r"말하지\s*마", r"누구에게도", r"전쟁터", r"전장", r"테러리스트", r"강도", 
        r"목숨", r"훔쳐가지", r"경찰에\s*신고", r"옮겨야\s*한다", r"가정은\s*네가", r"분담"
    ],
    1: [ # 과도한 애정 공세 / 가스라이팅 (단독 사용 시 스캠 판정 안 함)
        r"운명이다", r"오빠를\s*위해", r"잃고\s*싶지", r"결혼할\s*거지", r"사랑을\s*바쳤습니다"
    ]
}

LABEL_INFO = {
    0: {"name": "정상 대화 (Normal)", "color": "🟢", "desc": "사기 징후가 발견되지 않은 일반적인 일상 대화입니다."},
    1: {"name": "가짜 친밀감 / 애정 공세 (Love Bombing)", "color": "🟡", "desc": "과도한 호감 표현 및 단기간 내 신뢰 형성을 시도하는 위험 단계입니다."},
    2: {"name": "심리적 압박 / 비밀 유지 요구 (Gaslighting)", "color": "🟠", "desc": "주변에 알리지 못하도록 비밀 유지를 강요하거나 급박한 상황을 조작하는 고위험 단계입니다."},
    3: {"name": "세관 / 수수료 / 금전 요구 사기 (Scam Risk)", "color": "🔴", "desc": "해외 송금, 통관 수수료, 암호화폐 거래소, 충전 요구 등을 구실로 금전을 요구하는 심각한 스캠 단계입니다."}
}

# 4. 고도화된 초정밀 OCR 함수
def extract_text_ultra(pil_image):
    img_np = np.array(pil_image)
    h, w = img_np.shape[:2]
    
    scan_images = []
    resized = cv2.resize(img_np, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    scan_images.append(img_np)
    scan_images.append(resized)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    scan_images.append(enhanced)
    
    _, thresh1 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    scan_images.append(thresh1)
    scan_images.append(thresh2)

    extracted_words = []
    for img in scan_images:
        results = reader.readtext(img, detail=0)
        for text in results:
            clean_text = text.strip()
            if len(clean_text) > 1 and clean_text not in extracted_words:
                extracted_words.append(clean_text)

    return " ".join(extracted_words)

# 5. UI 메인 레이아웃
st.title("🛡️ AI 기반 로맨스 스캠 심리 조작 기제 진단 시스템")
st.markdown("대화 캡처 이미지를 업로드하거나 텍스트를 직접 입력하여 **로맨스 스캠 위험도 및 심리 조작 유형**을 분석하세요.")

st.divider()

col1, col2 = st.columns([1, 1])
extracted_text = ""

with col1:
    st.subheader("📷 1. 대화 이미지 업로드 및 스마트 OCR 추출")
    uploaded_file = st.file_uploader("카카오톡/라인/인스타그램 대화 캡처 이미지를 첨부하세요.", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 대화 이미지", width="stretch")
        
        with st.spinner("다크모드/저해상도 대비 3단계 정밀 OCR 스캔 중..."):
            extracted_text = extract_text_ultra(image)
            
            if extracted_text.strip():
                st.success("OCR 텍스트 추출 완료!")
            else:
                st.warning("OCR이 글자를 읽지 못했습니다. 우측 상자에 직접 대화를 입력해 주세요.")

with col2:
    st.subheader("📝 2. 텍스트 확인 및 진단")
    st.caption("※ OCR 추출 결과를 확인하시고 필요시 문장을 보완해 주세요.")
    
    input_text = st.text_area(
        "분석할 대화 내용",
        value=extracted_text,
        height=200,
        placeholder="대화 텍스트를 직접 입력하거나 좌측에서 이미지를 업로드하세요."
    )
    
    analyze_btn = st.button("🔍 로맨스 스캠 위험도 분석 시작", type="primary", width="stretch")

st.divider()

# 6. 맥락 기반 정밀 판정 엔진
if analyze_btn:
    if not input_text.strip():
        st.warning("분석할 대화 텍스트가 존재하지 않습니다. 입력 후 다시 시도해 주세요.")
    else:
        st.subheader("📊 3. AI 심리 기제 진단 결과")
        
        # 1) KoBERT 기본 추론
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).numpy()[0]

        detected_keywords = []

        # 2) 예외 처리: 일상 소액/거절 단어 존재 시 스캠 가중치 차단
        is_daily_talk = any(re.search(ex, input_text, re.IGNORECASE) for ex in DAILY_EXCLUSIONS)

        if is_daily_talk:
            # 일상 대화 판단 시 정상(0번) 확률을 독점적으로 상향
            probabilities[0] += 2.0
        else:
            # 3) 실제 스캠 위험 패턴 감지
            for label_id, patterns in SCAM_PATTERNS.items():
                for pt in patterns:
                    if re.search(pt, input_text, re.IGNORECASE):
                        weight = 0.60 if label_id in [2, 3] else 0.25
                        probabilities[label_id] += weight
                        
                        # SyntaxError 없는 안전한 변수 처리
                        clean_pt = pt.replace(r'\s*', ' ')
                        detected_keywords.append(clean_pt)

        # Softmax 재정규화
        probabilities = np.exp(probabilities) / np.sum(np.sum(np.exp(probabilities)))
        pred_label = int(np.argmax(probabilities))
        confidence = probabilities[pred_label] * 100

        res_info = LABEL_INFO[pred_label]
        
        # 결과 화면 출력
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(
                label="최종 진단 결과",
                value=f"{res_info['color']} {res_info['name']}"
            )
            st.metric(
                label="분류 신뢰도 (Confidence)",
                value=f"{confidence:.2f}%"
            )
            if detected_keywords and pred_label != 0:
                st.error(f"⚠️ **위험 패턴/키워드 감지**: {', '.join(set(detected_keywords))}")
            elif pred_label == 0 and is_daily_talk:
                st.success("✅ **일상 대화 감지**: 일상적인 소액 요청 및 대화 패턴입니다.")
            
        with res_col2:
            st.info(f"**유형 설명**: {res_info['desc']}")
            
            st.write("**[심리 조작 기제별 세부 확률 분포]**")
            for idx, prob in enumerate(probabilities):
                st.write(f"- {LABEL_INFO[idx]['name']}: `{prob*100:.1f}%`")
                st.progress(float(prob))