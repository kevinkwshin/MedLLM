import streamlit as st
import pandas as pd
import re
import time
from io import StringIO
from openai import OpenAI
from datetime import datetime
import json

# --- 페이지 설정 ---
st.set_page_config(
    page_title="🏥 MedLLM Benchmark",
    page_icon="🏥",
    layout="wide",
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
    .benchmark-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        color: #333;
    }
    .benchmark-card h4 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .benchmark-card p {
        color: #555;
        margin: 0.3rem 0;
    }
    .accuracy-high { color: #28a745; font-weight: bold; }
    .accuracy-medium { color: #ffc107; font-weight: bold; }
    .accuracy-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 세션 상태 초기화 ---
if 'benchmark_results' not in st.session_state:
    # 샘플 벤치마크 결과 생성
    st.session_state.benchmark_results = [
        {
            'id': 1,
            'model_name': 'Qwen/Qwen3-32B',
            'accuracy': 87.5,
            'total_questions': 240,
            'correct_answers': 210,
            'evaluation_date': '2025-01-25 14:30:22',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 2
        },
        {
            'id': 2,
            'model_name': 'Qwen/Qwen3-1.7B',
            'accuracy': 82.3,
            'total_questions': 240,
            'correct_answers': 197,
            'evaluation_date': '2025-01-24 16:45:11',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 0
        },
        {
            'id': 3,
            'model_name': 'google/gemma-2-27b-it',
            'accuracy': 79.6,
            'total_questions': 240,
            'correct_answers': 191,
            'evaluation_date': '2025-01-23 09:15:33',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 1
        },
        {
            'id': 4,
            'model_name': 'microsoft/DialoGPT-medium',
            'accuracy': 71.2,
            'total_questions': 240,
            'correct_answers': 171,
            'evaluation_date': '2025-01-22 11:20:45',
            'dataset_name': 'Medical QA Dataset v1.1',
            'api_errors': 8
        },
        {
            'id': 5,
            'model_name': 'anthropic/claude-3-haiku',
            'accuracy': 91.7,
            'total_questions': 240,
            'correct_answers': 220,
            'evaluation_date': '2025-01-21 13:50:17',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 0
        }
    ]

if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False

if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

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
        api_errors = 0

        for i, (question, correct_answer) in enumerate(zip(questions, correct_answers)):
            prompt = f"""{question}\n\nAnswer with the letter of the correct option (A, B, C, or D).\n\nAnswer:"""
            
            response_text = self.query_model(model_name, prompt)
            
            predicted_answer = "EXTRACT_FAIL"
            error_message = None

            if response_text.startswith("API_ERROR:"):
                error_message = response_text
                predicted_answer = "API_ERROR"
                api_errors += 1
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

        return results, api_errors

def get_accuracy_class(accuracy):
    if accuracy >= 85:
        return "accuracy-high"
    elif accuracy >= 70:
        return "accuracy-medium"
    else:
        return "accuracy-low"

def add_benchmark_result(model_name, accuracy, total_questions, correct_answers, dataset_name, api_errors):
    new_id = max([r['id'] for r in st.session_state.benchmark_results]) + 1 if st.session_state.benchmark_results else 1
    new_result = {
        'id': new_id,
        'model_name': model_name,
        'accuracy': accuracy,
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_name': dataset_name,
        'api_errors': api_errors
    }
    st.session_state.benchmark_results.append(new_result)

def delete_benchmark_result(result_id):
    st.session_state.benchmark_results = [r for r in st.session_state.benchmark_results if r['id'] != result_id]

# --- 메인 헤더 ---
st.markdown('<div class="main-header"><h1>🏥 MedLLM Benchmark Results</h1></div>', unsafe_allow_html=True)

# --- 관리자 모드 토글 (숨김) ---
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button("🔧", help="Admin Mode"):
        if st.session_state.admin_authenticated:
            # 이미 인증된 경우 모드 토글
            st.session_state.admin_mode = not st.session_state.admin_mode
            if not st.session_state.admin_mode:
                st.session_state.admin_authenticated = False
        else:
            # 인증이 필요한 경우
            st.session_state.admin_mode = True

# --- 관리자 인증 섹션 ---
if st.session_state.admin_mode and not st.session_state.admin_authenticated:
    st.markdown("---")
    st.header("🔐 Admin Authentication Required")
    
    col_auth1, col_auth2, col_auth3 = st.columns([1, 2, 1])
    with col_auth2:
        password = st.text_input("Enter Admin Password:", type="password", key="admin_password")
        
        col_login, col_cancel = st.columns(2)
        with col_login:
            if st.button("🔑 Login", key="admin_login", use_container_width=True):
                if password == "passpass":
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin 인증 성공!")
                    st.rerun()
                else:
                    st.error("❌ 잘못된 비밀번호입니다.")
        
        with col_cancel:
            if st.button("❌ Cancel", key="admin_cancel", use_container_width=True):
                st.session_state.admin_mode = False
                st.rerun()

# --- 벤치마크 결과 표시 ---
st.header("📊 Current Benchmark Rankings")

# 정확도 순으로 정렬
sorted_results = sorted(st.session_state.benchmark_results, key=lambda x: x['accuracy'], reverse=True)

for i, result in enumerate(sorted_results):
    accuracy_class = get_accuracy_class(result['accuracy'])
    
    col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
    
    with col1:
        st.markdown(f"<h2>#{i+1}</h2>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="benchmark-card">
            <h4>{result['model_name']}</h4>
            <p><strong>Dataset:</strong> {result['dataset_name']}</p>
            <p><strong>Date:</strong> {result['evaluation_date']}</p>
            {f"<p><strong>API Errors:</strong> {result['api_errors']}</p>" if result['api_errors'] > 0 else ""}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="text-align: center;">
            <h2 class="{accuracy_class}">{result['accuracy']:.1f}%</h2>
            <p>{result['correct_answers']} / {result['total_questions']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.admin_mode and st.session_state.admin_authenticated:
            if st.button("🗑️", key=f"delete_{result['id']}", help="Delete"):
                delete_benchmark_result(result['id'])
                st.rerun()

# --- 관리자 모드: 새로운 평가 실행 ---
if st.session_state.admin_mode and st.session_state.admin_authenticated:
    st.markdown("---")
    st.header("🔧 Admin: Run New Evaluation")

    with st.sidebar:
        st.header("⚙️ Evaluation Settings")

        st.info("**권장:** Streamlit Cloud의 Secrets에 `HF_TOKEN`을 설정하세요.")
        api_key = st.text_input(
            "Hugging Face Token (hf_...)", 
            type="password", 
            help="Hugging Face API 토큰을 입력하세요.",
            value=st.secrets.get("HF_TOKEN", "")
        )
        
        # 여러 모델 입력
        model_ids_input = st.text_area(
            "HuggingFace Model IDs (한 줄에 하나씩)", 
            value="Qwen/Qwen3-32BQwen/Qwen3-1.7B/Llama-3.1-7B-Instruct",
            help="평가할 모델들을 한 줄에 하나씩 입력하세요.",
            height=150
        )

        dataset_name = st.text_input(
            "Dataset Name",
            value="Medical QA Dataset v1.3",
            help="데이터셋 이름을 입력하세요."
        )

        st.subheader("📊 데이터셋")
        uploaded_file = st.file_uploader(
            "CSV 파일을 업로드하세요",
            type=['csv'],
            help="'question'과 'answer' 컬럼을 포함해야 합니다."
        )
        
        sample_csv = '''question,answer
"25세 여성이 심한 복통으로 응급실에 내원했습니다. 오른쪽 하복부에 압통이 있고, 발열과 오심을 동반합니다. 가장 가능성이 높은 진단은?\nA) 급성 위염\nB) 급성 충수염\nC) 요로감염\nD) 급성 담낭염","B"
"45세 남성이 운동 후 심한 흉통을 호소합니다. 흉통은 왼쪽 팔로 방사되며, 식은땀을 흘리고 있습니다. 가장 우선적으로 시행해야 할 검사는?\nA) 흉부 X-ray\nB) 심전도(ECG)\nC) 복부 CT\nD) 혈액검사","B"
"60세 여성이 3개월간의 체중감소와 복부팽만을 주소로 내원했습니다. 복부 초음파에서 복강 내 다량의 복수가 관찰됩니다. 가장 가능성이 높은 원인은?\nA) 심부전\nB) 간경화\nC) 복막염\nD) 난소암","D"
'''
        st.download_button(
            label="📥 샘플 CSV 다운로드",
            data=sample_csv,
            file_name="medical_qa_sample.csv",
            mime="text/csv"
        )

    if st.button("🚀 Start Batch Evaluation"):
        if not api_key:
            st.error("❌ Hugging Face 토큰을 입력하거나 Secrets에 설정하세요.")
        elif not model_ids_input.strip():
            st.error("❌ 모델 ID를 입력하세요.")
        elif uploaded_file is None:
            st.error("❌ 데이터셋 파일을 업로드하세요.")
        else:
            try:
                # 모델 IDs 파싱
                model_ids = [mid.strip() for mid in model_ids_input.strip().split('\n') if mid.strip()]
                
                df = pd.read_csv(uploaded_file)
                if 'question' not in df.columns or 'answer' not in df.columns:
                    st.error("❌ CSV 파일에 'question'과 'answer' 컬럼이 있어야 합니다.")
                else:
                    st.info(f"✅ {len(df)}개의 질문으로 데이터셋 로드 완료.")
                    st.info(f"🤖 {len(model_ids)}개 모델 순차 평가 시작...")

                    evaluator = MedicalEvaluator(api_key)
                    questions = df['question'].tolist()
                    correct_answers = df['answer'].tolist()

                    # 각 모델에 대해 순차적으로 평가
                    for model_idx, model_id in enumerate(model_ids):
                        clean_model_id = model_id.strip().strip('"\'')
                        
                        st.subheader(f"Evaluating Model {model_idx + 1}/{len(model_ids)}: {clean_model_id}")
                        
                        progress_bar = st.progress(0, text=f"평가 시작: {clean_model_id}")
                        evaluation_results, api_errors = evaluator.evaluate(clean_model_id, questions, correct_answers, progress_bar)
                        progress_bar.empty()

                        results_df = pd.DataFrame(evaluation_results)
                        correct_count = results_df['is_correct'].sum()
                        total_count = len(results_df)
                        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

                        # 벤치마크 결과에 추가
                        add_benchmark_result(
                            clean_model_id, 
                            accuracy, 
                            total_count, 
                            correct_count, 
                            dataset_name,
                            api_errors
                        )

                        st.success(f"✅ {clean_model_id}: {accuracy:.2f}% ({correct_count}/{total_count})")
                        
                        if api_errors > 0:
                            st.warning(f"⚠️ API 오류 {api_errors}개 발생")

                    st.success("🎉 모든 모델 평가 완료! 페이지가 새로고침됩니다.")
                    time.sleep(2)
                    st.rerun()

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

else:
    st.info("새로운 평가를 실행하려면 관리자에게 문의하세요.")
