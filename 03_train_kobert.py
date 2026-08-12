"""
03_train_kobert.py (최신 transformers v5 API 호환 수정본)
- Trainer의 processing_class 인자 적용
"""

import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

def train_kobert():
    print("="*60)
    print("🚀 [3단계] KoBERT 모델 파인튜닝 학습 시작")
    print("="*60)

    # 1. 데이터셋 로드
    csv_path = os.path.join('data', 'romance_scam_dataset.csv')
    if not os.path.exists(csv_path):
        print("❌ 데이터셋 파일이 없습니다. 02_create_dataset.py를 먼저 실행해 주세요.")
        return

    df = pd.read_csv(csv_path)
    
    # 학습/검증 데이터 분할 (80% 학습, 20% 검증)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # HuggingFace Dataset 객체로 변환
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # 2. 토크나이저 및 사전학습 한국어 BERT 모델 로드
    model_name = "kykim/bert-kor-base"
    print(f"\n🔄 베이스 모델 및 토크나이저 로드 중: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # 3. 텍스트 토큰화 전처리 함수
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)

    print("⚡ 데이터 토큰화 수행 중...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)

    # 4. 학습 파라미터 설정
    output_dir = os.path.join('models', 'kobert_romance_scam')
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,                     # Epoch 수
        per_device_train_batch_size=4,          # 학습 배치 사이즈
        per_device_eval_batch_size=4,           # 평가 배치 사이즈
        learning_rate=2e-5,                     # 파인튜닝 권장 학습률
        weight_decay=0.01,                      # 과적합 방지 규제
        save_strategy="epoch",                  # 에포크마다 모델 저장
        use_cpu=True                            # CPU 환경 안정성 보장
    )

    # 5. Trainer 객체 생성 및 학습 진행 (processing_class=tokenizer로 변경)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("\n🏋️‍♂️ 모델 학습(Training) 시작...")
    train_result = trainer.train()

    # 6. 완성된 모델 및 토크나이저 저장
    print(f"\n💾 학습 완료된 모델 저장 중... ({output_dir})")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("="*60)
    print("🎉 KoBERT 심리 조작 기제 분류 모델 학습 완료!")
    print("="*60)

if __name__ == "__main__":
    train_kobert()