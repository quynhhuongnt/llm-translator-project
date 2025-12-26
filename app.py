import streamlit as st
import google.generativeai as genai
import easyocr
from PIL import Image
import numpy as np
import docx
import PyPDF2
from io import StringIO

# 1. CẤU HÌNH TRANG & CSS (Giữ nguyên giao diện của bạn)
st.set_page_config(page_title="LLM Cloud Translator", layout="wide", page_icon="🌐")

st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px; height: 300px; font-family: sans-serif; }
    .stButton button { 
        background-color: #1a73e8; color: white; font-size: 16px; 
        border-radius: 8px; padding: 0.5rem 1rem; border: none; 
        width: 100%; font-weight: bold;
    }
    .stButton button:hover { background-color: #1557b0; color: white; }
    .result-box { 
        border: 1px solid #d3d3d3; border-radius: 0.5rem; padding: 1rem;             
        height: 300px; background-color: #f0f2f6; color: #31333F;           
        overflow-y: auto; font-family: sans-serif; font-size: 16px;          
        white-space: pre-wrap; user-select: text; cursor: text;               
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .lang-header { font-weight: bold; font-size: 18px; margin-bottom: 10px; display: block; color: #1a73e8; }
</style>
""", unsafe_allow_html=True)

# 2. BACKEND LOGIC
@st.cache_resource
def load_ocr():
    # Thêm tham số gpu=False vì Streamlit Cloud không có GPU miễn phí
    return easyocr.Reader(['en'], gpu=False)

# Khởi tạo model bên ngoài hàm để tránh gọi lại nhiều lần
def get_gemini_model():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        # Sửa lỗi NotFound bằng cách chỉ định chính xác phiên bản
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Lỗi cấu hình API: {e}")
        return None

model = get_gemini_model()
reader = load_ocr()

# Tối ưu hàm dịch: Sử dụng streaming để tránh timeout trên Cloud
def translate_with_llm(text, type_context="văn bản"):
    if not text or not text.strip(): return ""
    prompt = f"Bạn là chuyên gia dịch thuật. Hãy dịch đoạn {type_context} sau sang tiếng Việt một cách tự nhiên: \n\n{text}"
    
    try:
        # Sử dụng stream=True để nhận dữ liệu liên tục, tránh bị đứng app
        response = model.generate_content(prompt, stream=True)
        full_text = ""
        # Tạo placeholder để hiện chữ chạy dần dần (UX tốt hơn)
        with st.empty():
            for chunk in response:
                full_text += chunk.text
                # Chỉ hiển thị tạm thời ở đây nếu bạn muốn hiệu ứng gõ chữ
        return full_text
    except Exception as e:
        return f"Lỗi dịch thuật: {str(e)}"

def read_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "text/plain":
            text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
        elif "word" in uploaded_file.type or "officedocument" in uploaded_file.type:
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
    return text

# 3. FRONTEND
st.title(" ỨNG DỤNG DỊCH ANH - VIỆT CLOUD LLM ")
st.markdown("Sinh viên thực hiện: Ngô Thị Quỳnh Hương | Mã SV: 99048")

tab_text, tab_image, tab_doc = st.tabs(["🔤 Văn Bản", "📸 Hình Ảnh", "📂 Tài Liệu"])

# Khởi tạo session state
for key in ['res_text', 'res_img', 'res_doc']:
    if key not in st.session_state: st.session_state[key] = ""

with tab_text:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="lang-header">TIẾNG ANH</span>', unsafe_allow_html=True)
        t_input = st.text_area("Input", height=300, label_visibility="collapsed", key="input_area")
    with c2:
        st.markdown('<span class="lang-header">TIẾNG VIỆT (AI)</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box">{st.session_state.res_text}</div>', unsafe_allow_html=True)
    
    if st.button("DỊCH NGAY", key="btn1"):
        if t_input:
            with st.spinner("Đang dịch..."):
                st.session_state.res_text = translate_with_llm(t_input)
                st.rerun()

with tab_image:
    c1, c2 = st.columns(2)
    with c1:
        img_file = st.file_uploader("Chọn ảnh", type=['png','jpg','jpeg'], label_visibility="collapsed")
        if img_file: st.image(Image.open(img_file), use_container_width=True)
    with c2:
        st.markdown(f'<div class="result-box">{st.session_state.res_img}</div>', unsafe_allow_html=True)
    
    if st.button("QUÉT & DỊCH", key="btn2"):
        if img_file:
            with st.spinner("Đang quét ảnh và dịch..."):
                img_np = np.array(Image.open(img_file))
                res_ocr = reader.readtext(img_np, detail=0)
                raw_ocr = " ".join(res_ocr)
                st.session_state.res_img = translate_with_llm(raw_ocr, "quét từ ảnh")
                st.rerun()

with tab_doc:
    doc_file = st.file_uploader("Chọn file", type=['pdf','docx','txt'], label_visibility="collapsed")
    if st.button("DỊCH TÀI LIỆU", key="btn3"):
        if doc_file:
            with st.spinner("Đang xử lý tài liệu..."):
                content = read_file(doc_file)
                st.session_state.res_doc = translate_with_llm(content, "tài liệu")
                st.rerun()
    
    if st.session_state.res_doc:
        st.markdown(f'<div class="result-box">{st.session_state.res_doc}</div>', unsafe_allow_html=True)
