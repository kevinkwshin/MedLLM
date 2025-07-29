import streamlit as st
import pandas as pd
import re
import time
from io import StringIO, BytesIO
from openai import OpenAI
from datetime import datetime
import json
import zipfile

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
        padding: 2.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* 제출 섹션 스타일링 */
    .submit-section {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #dee2e6;
        margin-bottom: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .submit-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .submit-info {
        color: #6c757d;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }
    
    /* 입력 필드와 버튼 정렬 */
    .input-container {
        display: flex;
        gap: 1rem;
        align-items: end;
    }
    
    .input-field {
        flex: 1;
    }
    
    .submit-button {
        flex-shrink: 0;
    }
    
    /* 버튼 스타일 개선 */
    .stButton>button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
        height: 56px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #218838 0%, #1ba085 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
    }
    
    /* 관리자 버튼 스타일 */
    .admin-button button {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        font-size: 1.2rem;
    }
    
    /* 벤치마크 카드 개선 */
    .benchmark-card {
        border: none;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #333;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .benchmark-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .benchmark-card h4 {
        color: #2c3e50;
        margin-bottom: 0.8rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .benchmark-card p {
        color: #6c757d;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    
    .benchmark-rank {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.5rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    
    .accuracy-score {
        text-align: center;
        padding: 1rem;
        background: rgba(255,255,255,0.8);
        border-radius: 12px;
        border: 2px solid #e9ecef;
    }
    
    .accuracy-high { 
        color: #28a745; 
        font-weight: 700;
        font-size: 2rem;
    }
    .accuracy-medium { 
        color: #ffc107; 
        font-weight: 700;
        font-size: 2rem;
    }
    .accuracy-low { 
        color: #dc3545; 
        font-weight: 700;
        font-size: 2rem;
    }
    
    /* 대기열 정보 스타일 */
    .queue-info {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* 섹션 헤더 스타일 */
    .section-header {
        color: #2c3e50;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# --- 세션 상태 초기화 ---
if 'benchmark_results' not in st.session_state:
    # 샘플 벤치마크 결과 생성 (Qwen3 30B 이하 모델들)
    st.session_state.benchmark_results = [
        {
            'id': 1,
            'model_name': 'Qwen/Qwen3-14B-Instruct',
            'accuracy': 86.7,
            'total_questions': 240,
            'correct_answers': 208,
            'evaluation_date': '2025-01-25 14:30:22',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 1
        },
        {
            'id': 2,
            'model_name': 'Qwen/Qwen3-7B-Instruct',
            'accuracy': 82.1,
            'total_questions': 240,
            'correct_answers': 197,
            'evaluation_date': '2025-01-24 16:45:11',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 2
        },
        {
            'id': 3,
            'model_name': 'Qwen/Qwen3-3B-Instruct',
            'accuracy': 77.5,
            'total_questions': 240,
            'correct_answers': 186,
            'evaluation_date': '2025-01-23 09:15:33',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 1
        },
        {
            'id': 4,
            'model_name': 'Qwen/Qwen3-1.5B-Instruct',
            'accuracy': 71.3,
            'total_questions': 240,
            'correct_answers': 171,
            'evaluation_date': '2025-01-22 11:20:45',
            'dataset_name': 'Medical QA Dataset v1.1',
            'api_errors': 4
        },
        {
            'id': 5,
            'model_name': 'Qwen/Qwen3-0.5B-Instruct',
            'accuracy': 63.8,
            'total_questions': 240,
            'correct_answers': 153,
            'evaluation_date': '2025-01-21 13:50:17',
            'dataset_name': 'Medical QA Dataset v1.2',
            'api_errors': 7
        }
    ]

if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False

if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

if 'test_dataset' not in st.session_state:
    # 보안 테스트 데이터셋 (실제로는 파일에서 로드)
    st.session_state.test_dataset = None

if 'pending_evaluations' not in st.session_state:
    st.session_state.pending_evaluations = []

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

def add_evaluation_request(model_name):
    """평가 요청을 대기열에 추가"""
    if model_name not in [req['model_name'] for req in st.session_state.pending_evaluations]:
        request_id = len(st.session_state.pending_evaluations) + 1
        st.session_state.pending_evaluations.append({
            'id': request_id,
            'model_name': model_name,
            'status': 'pending',
            'submitted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluated_at': None
        })
        return True
    return False

def load_secure_dataset(uploaded_file, password=None):
    """보안 테스트 데이터셋 로드 (암호화된 파일 지원)"""
    try:
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if file_extension == 'csv':
            # CSV 파일 처리
            if password:
                # 암호화된 ZIP 내의 CSV 처리
                try:
                    with zipfile.ZipFile(BytesIO(uploaded_file.read()), 'r') as zip_file:
                        zip_file.setpassword(password.encode())
                        # ZIP 내 첫 번째 CSV 파일 찾기
                        csv_files = [f for f in zip_file.namelist() if f.lower().endswith('.csv')]
                        if not csv_files:
                            return False, "ZIP 파일 내에 CSV 파일이 없습니다."
                        
                        with zip_file.open(csv_files[0]) as csv_file:
                            df = pd.read_csv(csv_file)
                except Exception as e:
                    return False, f"암호화된 ZIP 파일 읽기 실패: {str(e)}"
            else:
                # 일반 CSV 파일
                df = pd.read_csv(uploaded_file)
                
        elif file_extension in ['xlsx', 'xls']:
            # 엑셀 파일 처리
            try:
                if password:
                    return False, "암호화된 엑셀 파일을 처리하려면 msoffcrypto 라이브러리가 필요합니다. 현재 ZIP 암호화된 CSV 파일을 사용해주세요."
                else:
                    # 일반 엑셀 파일
                    df = pd.read_excel(uploaded_file)
                    
            except Exception as e:
                return False, f"엑셀 파일 읽기 실패: {str(e)}"
        else:
            return False, "지원되지 않는 파일 형식입니다. CSV, XLS, XLSX 파일만 지원됩니다."
        
        # 데이터 검증
        if 'question' not in df.columns or 'answer' not in df.columns:
            return False, "파일에 'question'과 'answer' 컬럼이 있어야 합니다."
        
        # 세션에 저장
        st.session_state.test_dataset = {
            'questions': df['question'].tolist(),
            'answers': df['answer'].tolist(),
            'total_count': len(df),
            'loaded_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_type': file_extension,
            'is_encrypted': password is not None
        }
        
        encryption_info = " (암호화됨)" if password else ""
        return True, f"{len(df)}개의 질문이 포함된 테스트 데이터셋이 로드되었습니다.{encryption_info}"
        
    except Exception as e:
        return False, f"데이터셋 로드 중 오류: {str(e)}"

def delete_benchmark_result(result_id):
    st.session_state.benchmark_results = [r for r in st.session_state.benchmark_results if r['id'] != result_id]

# --- 메인 헤더 ---
st.markdown('<div class="main-header"><h1>🏥 MedLLM Benchmark Results</h1></div>', unsafe_allow_html=True)

# --- 모델 제출 섹션 (공개) ---
st.markdown("""
<div class="submit-section">
    <div class="submit-header">
        🚀 Submit Your Model for Evaluation
    </div>
    <div class="submit-info">
        HuggingFace 모델 주소를 제출하면 자동으로 평가가 진행됩니다.
    </div>
</div>
""", unsafe_allow_html=True)

# 입력 필드와 버튼을 같은 높이에 배치
col_input, col_button = st.columns([4, 1])

with col_input:
    model_submission = st.text_input(
        "HuggingFace Model ID",
        placeholder="예: Qwen/Qwen3-7B-Instruct",
        help="평가하고 싶은 HuggingFace 모델의 전체 경로를 입력하세요.",
        label_visibility="collapsed"
    )

with col_button:
    # 빈 공간으로 버튼 위치 맞추기
    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
    submit_clicked = st.button("📤 Submit", use_container_width=True, type="primary")

# 제출 로직
if submit_clicked:
    if model_submission.strip():
        success = add_evaluation_request(model_submission.strip())
        if success:
            st.success(f"✅ {model_submission} 평가 요청이 제출되었습니다!")
            st.info("관리자가 승인하면 자동으로 평가가 진행됩니다.")
            st.rerun()
        else:
            st.warning("⚠️ 이미 제출된 모델입니다.")
    else:
        st.error("❌ 모델 ID를 입력해주세요.")

# --- 대기중인 평가 표시 ---
if st.session_state.pending_evaluations:
    pending_count = len([req for req in st.session_state.pending_evaluations if req['status'] == 'pending'])
    if pending_count > 0:
        st.markdown(f"""
        <div class="queue-info">
            🕐 현재 <strong>{pending_count}개</strong>의 모델이 평가 대기중입니다.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📋 대기중인 평가 목록 보기"):
            for req in st.session_state.pending_evaluations:
                if req['status'] == 'pending':
                    st.markdown(f"• **{req['model_name']}** (제출: {req['submitted_at']})")

st.markdown("---")

# --- 관리자 모드 토글 (숨김) ---
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    st.markdown('<div class="admin-button">', unsafe_allow_html=True)
    admin_clicked = st.button("🔧", help="Admin Mode")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if admin_clicked:
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
st.markdown('<h2 class="section-header">📊 Current Benchmark Rankings</h2>', unsafe_allow_html=True)

# 정확도 순으로 정렬
sorted_results = sorted(st.session_state.benchmark_results, key=lambda x: x['accuracy'], reverse=True)

for i, result in enumerate(sorted_results):
    accuracy_class = get_accuracy_class(result['accuracy'])
    
    col1, col2, col3, col4 = st.columns([1, 4, 2, 1])
    
    with col1:
        st.markdown(f'<div class="benchmark-rank">#{i+1}</div>', unsafe_allow_html=True)
    
    with col2:
        api_error_info = f"<p><strong>⚠️ API Errors:</strong> {result['api_errors']}</p>" if result['api_errors'] > 0 else ""
        st.markdown(f"""
        <div class="benchmark-card">
            <h4>🤖 {result['model_name']}</h4>
            <p><strong>📊 Dataset:</strong> {result['dataset_name']}</p>
            <p><strong>📅 Date:</strong> {result['evaluation_date']}</p>
            {api_error_info}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="accuracy-score">
            <div class="{accuracy_class}">{result['accuracy']:.1f}%</div>
            <p style="margin: 0.5rem 0 0 0; color: #6c757d; font-size: 0.9rem;">
                {result['correct_answers']} / {result['total_questions']} 정답
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.admin_mode and st.session_state.admin_authenticated:
            if st.button("🗑️", key=f"delete_{result['id']}", help="Delete Result"):
                delete_benchmark_result(result['id'])
                st.rerun()

# --- 관리자 모드: 평가 관리 ---
if st.session_state.admin_mode and st.session_state.admin_authenticated:
    st.markdown("---")
    st.header("🔧 Admin: Manage Evaluations")

    # 테스트 데이터셋 관리
    st.subheader("📊 Secure Test Dataset")
    if st.session_state.test_dataset:
        encryption_badge = "🔒 암호화됨" if st.session_state.test_dataset.get('is_encrypted', False) else "🔓 일반"
        file_type_badge = st.session_state.test_dataset.get('file_type', 'unknown').upper()
        
        st.success(f"✅ 테스트 데이터셋 로드됨: {st.session_state.test_dataset['total_count']}개 질문")
        st.info(f"📄 파일 형식: {file_type_badge} | 보안: {encryption_badge} | 로드 시간: {st.session_state.test_dataset['loaded_at']}")
        
        col_ds1, col_ds2 = st.columns(2)
        with col_ds1:
            if st.button("🗑️ Remove Dataset", help="현재 로드된 테스트 데이터셋을 제거합니다."):
                st.session_state.test_dataset = None
                st.success("데이터셋이 제거되었습니다.")
                st.rerun()
        
        with col_ds2:
            if st.button("🔄 Reload Dataset", help="테스트 데이터셋을 다시 로드합니다."):
                st.session_state.test_dataset = None
                st.rerun()
    else:
        st.warning("⚠️ 테스트 데이터셋이 로드되지 않았습니다.")
        
        # 파일 업로드 및 암호 입력
        uploaded_test_file = st.file_uploader(
            "보안 테스트 데이터셋 업로드",
            type=['csv', 'xlsx', 'xls'],
            help="암호화된 파일도 지원됩니다. CSV, Excel 파일 모두 가능합니다.",
            key="secure_dataset"
        )
        
        # 암호 입력 (선택사항)
        dataset_password = st.text_input(
            "파일 암호 (선택사항)",
            type="password",
            help="암호화된 파일의 경우 암호를 입력하세요. 일반 파일은 비워두세요.",
            key="dataset_password"
        )
        
        col_upload1, col_upload2 = st.columns(2)
        
        with col_upload1:
            if st.button("📁 Load Dataset", help="데이터셋을 로드합니다."):
                if uploaded_test_file:
                    success, message = load_secure_dataset(uploaded_test_file, dataset_password if dataset_password else None)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("파일을 먼저 선택해주세요.")
        
        with col_upload2:
            st.info("💡 **지원 형식:**\n- CSV 파일 (일반/ZIP 암호화)\n- Excel 파일 (일반)\n- 'question', 'answer' 컬럼 필수")

    # 대기중인 평가 관리
    st.subheader("📋 Pending Evaluation Queue")
    if st.session_state.pending_evaluations:
        pending_requests = [req for req in st.session_state.pending_evaluations if req['status'] == 'pending']
        
        if pending_requests:
            st.info(f"🕐 {len(pending_requests)}개의 평가 요청이 대기중입니다.")
            
            for req in pending_requests:
                col_req1, col_req2, col_req3 = st.columns([3, 1, 1])
                
                with col_req1:
                    st.text(f"📦 {req['model_name']}")
                    st.caption(f"제출: {req['submitted_at']}")
                
                with col_req2:
                    if st.button("✅ Approve", key=f"approve_{req['id']}", help="평가 승인 및 실행"):
                        if st.session_state.test_dataset:
                            req['status'] = 'approved'
                            st.success("평가가 승인되었습니다!")
                            st.rerun()
                        else:
                            st.error("❌ 테스트 데이터셋을 먼저 로드해주세요.")
                
                with col_req3:
                    if st.button("❌ Reject", key=f"reject_{req['id']}", help="평가 요청 거부"):
                        st.session_state.pending_evaluations.remove(req)
                        st.success("평가 요청이 거부되었습니다.")
                        st.rerun()
                
                st.markdown("---")
        else:
            st.info("처리할 평가 요청이 없습니다.")
    else:
        st.info("제출된 평가 요청이 없습니다.")

    # 수동 평가 섹션
    st.subheader("🔧 Manual Evaluation")
    
    with st.sidebar:
        st.header("⚙️ Manual Evaluation Settings")

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
            value="Qwen/Qwen3-14B-Instruct\nQwen/Qwen3-7B-Instruct",
            help="평가할 모델들을 한 줄에 하나씩 입력하세요.",
            height=150
        )

        dataset_name = st.text_input(
            "Dataset Name",
            value="Medical QA Dataset v1.3",
            help="데이터셋 이름을 입력하세요."
        )

    if st.button("🚀 Start Manual Evaluation"):
        if not api_key:
            st.error("❌ Hugging Face 토큰을 입력하거나 Secrets에 설정하세요.")
        elif not model_ids_input.strip():
            st.error("❌ 모델 ID를 입력하세요.")
        elif not st.session_state.test_dataset:
            st.error("❌ 테스트 데이터셋을 먼저 로드해주세요.")
        else:
            try:
                # 모델 IDs 파싱
                model_ids = [mid.strip() for mid in model_ids_input.strip().split('\n') if mid.strip()]
                
                questions = st.session_state.test_dataset['questions']
                correct_answers = st.session_state.test_dataset['answers']
                
                st.info(f"🤖 {len(model_ids)}개 모델 순차 평가 시작...")

                evaluator = MedicalEvaluator(api_key)

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
    st.info("🔧 관리자 기능을 사용하려면 우측 상단의 설정 버튼을 클릭하세요.")
