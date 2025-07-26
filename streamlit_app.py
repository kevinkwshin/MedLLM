import streamlit as st
import pandas as pd
import requests
import re
import time
from io import StringIO

# --- 페이지 설정 ---
st.set_page_config(
    page_title="🏥 MedLLM Simple Evaluator",
    page_icon="🏥",
    layout="centered",
)

# --- CSS 스타일링 ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- HuggingFace API 클래스 ---
class MedicalQA:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = 'https://api-inference.huggingface.co/models/'
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        }

    def query_model(self, model_name, prompt):
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 10,
                "temperature": 0.1,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True}
        }
        try:
            response = requests.post(
                f"{self.api_url}{model_name}",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API Error: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def evaluate(self, model_name, questions, correct_answers, progress_bar):
        results = []
        total_questions = len(questions)

        for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
            prompt = f"""{question}

Answer with the letter of the correct option (A, B, C, or D).

Answer:"""
            
            response = self.query_model(model_name, prompt)
            
            predicted_answer = "ERROR"
            raw_response = str(response)

            if isinstance(response, list) and 'generated_text' in response[0]:
                generated_text = response[0]['generated_text'].strip()
                raw_response = generated_text
                # A, B, C, D 중 하나를 추출
                match = re.search(r'\b([A-D])\b', generated_text.upper())
                if match:
                    predicted_answer = match.group(1)

            is_correct = predicted_answer == correct_answer
            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'raw_response': raw_response
            })
            
            progress_bar.progress((i + 1) / total_questions)
            time.sleep(1) # API 요청 제한을 피하기 위함

        return results

# --- 메인 앱 ---

# 헤더
st.markdown('<div class="main-header"><h1>🏥 MedLLM Simple Evaluator</h1></div>', unsafe_allow_html=True)

# --- 입력을 위한 사이드바 ---
with st.sidebar:
    st.header("⚙️ 설정")

    # 1. API 키
    api_key = st.text_input("HuggingFace API Key", type="password", help="HuggingFace API 키를 입력하세요.")
    
    # 2. 모델 선택
    model_id = st.text_input("HuggingFace Model ID", "google/gemma-2b-it", help="예: 'microsoft/BioGPT', 'google/gemma-2b-it'")

    # 3. 데이터셋 업로드
    st.subheader("📊 데이터셋")
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요",
        type=['csv'],
        help="CSV 파일은 'question'과 'answer' 컬럼을 포함해야 합니다."
    )
    
    # 샘플 CSV 다운로드
    simplified_sample = '''question,answer
"25세 여성이 심한 복통으로 응급실에 내원했습니다. 마지막 월경은 6주 전이었고, 소변 임신 검사 결과는 양성입니다. 혈압은 90/60 mmHg, 맥박은 분당 110회입니다. 검사 결과 좌하복부에 압통이 있습니다. 가장 가능성이 높은 진단은 무엇입니까?\n\nA. 자궁외 임신\nB. 맹장염\nC. 난소 낭종 파열\nD. 골반 염증성 질환","A"
"45세 남성이 운동 후 심한 흉통을 호소합니다. 통증은 왼쪽 어깨와 팔로 방사되며, 식은땀을 흘리고 있습니다. 심전도상 V2-V4 유도에서 ST 분절 상승이 관찰됩니다. 가장 적절한 초기 치료는 무엇입니까?\n\nA. 아스피린\nB. 설하 니트로글리세린\nC. 산소\nD. 즉각적인 심장 도관술","D"
'''
    st.download_button(
        label="📥 샘플 CSV 다운로드",
        data=simplified_sample,
        file_name="medical_qa_sample.csv",
        mime="text/csv"
    )

# --- 평가 로직 ---
if st.button("🚀 평가 시작"):
    # 입력 값 검증
    if not api_key:
        st.error("❌ HuggingFace API 키를 입력하세요.")
    elif not model_id:
        st.error("❌ 모델 ID를 입력하세요.")
    elif uploaded_file is None:
        st.error("❌ 데이터셋 파일을 업로드하세요.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
            if 'question' not in df.columns or 'answer' not in df.columns:
                st.error("❌ CSV 파일에 'question'과 'answer' 컬럼이 있어야 합니다.")
            else:
                st.info(f"✅ {len(df)}개의 질문으로 데이터셋 로드 완료.")
                st.info(f"🤖 모델 평가 중: {model_id}")

                evaluator = MedicalQA(api_key)
                
                # 데이터 준비
                questions = df['question'].tolist()
                correct_answers = df['answer'].tolist()

                # 평가 실행
                progress_bar = st.progress(0)
                evaluation_results = evaluator.evaluate(model_id, questions, correct_answers, progress_bar)
                progress_bar.progress(1.0)
                st.success("🎉 평가 완료!")

                # --- 결과 표시 ---
                correct_count = sum(1 for r in evaluation_results if r['is_correct'])
                total_count = len(evaluation_results)
                accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

                st.metric("🏆 최종 정확도", f"{accuracy:.2f}%", f"{correct_count} / {total_count} 정답")

                # 상세 결과를 확장 가능한 형태로 표시
                with st.expander("📄 상세 결과 보기"):
                    results_df = pd.DataFrame(evaluation_results)
                    st.dataframe(results_df)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

else:
    st.info("사이드바에서 설정을 완료한 후 '평가 시작' 버튼을 클릭하세요.")
