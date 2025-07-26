import streamlit as st
import pandas as pd
import re
import time
from io import StringIO
from openai import OpenAI

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

# --- HuggingFace OpenAI 호환 API 클래스 ---
class MedicalEvaluator:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Hugging Face API 키가 필요합니다.")
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )

    def query_model(self, model_name, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"API_ERROR: {str(e)}"

    def evaluate(self, model_name, questions, correct_answers, progress_bar):
        results = []
        total_questions = len(questions)

        for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
            prompt = f"""{question}\n\nAnswer with the letter of the correct option (A, B, C, or D).\n\nAnswer:"""
            
            response_text = self.query_model(model_name, prompt)
            
            predicted_answer = "EXTRACT_FAIL"
            error_message = None

            if response_text.startswith("API_ERROR:"):
                error_message = response_text
                predicted_answer = "API_ERROR"
            else:
                match = re.search(r'\b([A-D])\b', response_text.upper())
                if match:
                    predicted_answer = match.group(1)

            is_correct = predicted_answer == correct_answer
            results.append({
                'question': question,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'raw_response': response_text,
                'error': error_message
            })
            
            progress_bar.progress((i + 1) / total_questions, text=f"Processing {i+1}/{total_questions}")
            time.sleep(0.2)

        return results

# --- 메인 앱 ---
st.markdown('<div class="main-header"><h1>🏥 MedLLM Simple Evaluator</h1></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 설정")

    st.info("**권장:** Streamlit Cloud의 Secrets에 `HF_TOKEN`을 설정하세요.")
    api_key = st.text_input(
        "Hugging Face Token (hf_...)", 
        type="password", 
        help="Hugging Face API 토큰을 입력하세요.",
        value=st.secrets.get("HF_TOKEN", "")
    )
    
    # 기본 모델을 Qwen으로 변경
    model_id = st.text_input(
        "HuggingFace Model ID", 
        "Qwen/Qwen3-Coder-480B-A35B-Instruct:novita", 
        help="예: 'Qwen/Qwen3-Coder-480B-A35B-Instruct:novita', 'google/gemma-2b-it'"
    )

    st.subheader("📊 데이터셋")
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요",
        type=['csv'],
        help="'question'과 'answer' 컬럼을 포함해야 합니다."
    )
    
    sample_csv = '''question,answer
"25세 여성이 심한 복통으로 응급실에 내원했습니다...","A"
"45세 남성이 운동 후 심한 흉통을 호소합니다...","D"
'''
    st.download_button(
        label="📥 샘플 CSV 다운로드",
        data=sample_csv,
        file_name="medical_qa_sample.csv",
        mime="text/csv"
    )

if st.button("🚀 평가 시작"):
    if not api_key:
        st.error("❌ Hugging Face 토큰을 입력하거나 Secrets에 설정하세요.")
    elif not model_id:
        st.error("❌ 모델 ID를 입력하세요.")
    elif uploaded_file is None:
        st.error("❌ 데이터셋 파일을 업로드하세요.")
    else:
        try:
            # 입력된 모델 ID의 공백 및 따옴표 제거
            clean_model_id = model_id.strip().strip('"\'')

            df = pd.read_csv(uploaded_file)
            if 'question' not in df.columns or 'answer' not in df.columns:
                st.error("❌ CSV 파일에 'question'과 'answer' 컬럼이 있어야 합니다.")
            else:
                st.info(f"✅ {len(df)}개의 질문으로 데이터셋 로드 완료.")
                st.info(f"🤖 모델 평가 중: {clean_model_id}")

                evaluator = MedicalEvaluator(api_key)
                questions = df['question'].tolist()
                correct_answers = df['answer'].tolist()

                progress_bar = st.progress(0, text="평가를 시작합니다...")
                evaluation_results = evaluator.evaluate(clean_model_id, questions, correct_answers, progress_bar)
                progress_bar.empty()
                st.success("🎉 평가 완료!")

                results_df = pd.DataFrame(evaluation_results)
                correct_count = results_df['is_correct'].sum()
                total_count = len(results_df)
                accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

                st.metric("🏆 최종 정확도", f"{accuracy:.2f}%", f"{correct_count} / {total_count} 정답")

                errors = results_df[results_df['error'].notna()]
                if not errors.empty:
                    st.warning(f"⚠️ {len(errors)}개의 질문에서 API 오류가 발생했습니다.")
                    top_error = errors['error'].value_counts().index[0]
                    st.code(top_error, language=None)

                with st.expander("📄 상세 결과 보기"):
                    st.dataframe(results_df)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")

else:
    st.info("사이드바에서 설정을 완료한 후 '평가 시작' 버튼을 클릭하세요.")
