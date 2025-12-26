import streamlit as st
import google.generativeai as genai
import easyocr
from PIL import Image
import numpy as np
import docx
import PyPDF2
from io import StringIO

# 1. CẤU HÌNH TRANG & CSS (Giữ nguyên phong cách của bạn)
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
    return easyocr.Reader(['en'])

# Cấu hình API Gemini
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error("Chưa cấu hình API Key trong Secrets!")

reader = load_ocr()

def translate_with_llm(text, type_context="văn bản"):
    if not text.strip(): return ""
    prompt = f"Bạn là chuyên gia dịch thuật. Hãy dịch đoạn {type_context} sau sang tiếng Việt một cách tự nhiên: {text}"
    response = model.generate_content(prompt)
    return response.text

def read_file(uploaded_file):
    text = ""
    if uploaded_file.type == "text/plain":
        text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    elif "word" in uploaded_file.type:
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# 3. FRONTEND (Giao diện theo yêu cầu của bạn)
st.title(" ỨNG DỤNG DỊCH ANH - VIỆT CLOUD LLM ")
st.markdown("Sinh viên thực hiện: Ngô Thị Quỳnh Hương | Mã SV: 99048")

tab_text, tab_image, tab_doc = st.tabs(["🔤 Văn Bản", "📸 Hình Ảnh", "📂 Tài Liệu"])

if 'res_text' not in st.session_state: st.session_state.res_text = ""
if 'res_img' not in st.session_state: st.session_state.res_img = ""
if 'res_doc' not in st.session_state: st.session_state.res_doc = ""

with tab_text:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="lang-header">TIẾNG ANH</span>', unsafe_allow_html=True)
        t_input = st.text_area("Input", height=300, label_visibility="collapsed")
    with c2:
        st.markdown('<span class="lang-header">TIẾNG VIỆT (AI)</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box">{st.session_state.res_text}</div>', unsafe_allow_html=True)
    
    if st.button("DỊCH NGAY", key="btn1"):
        st.session_state.res_text = translate_with_llm(t_input)
        st.rerun()

with tab_image:
    c1, c2 = st.columns(2)
    with c1:
        img_file = st.file_uploader("Chọn ảnh", type=['png','jpg','jpeg'], label_visibility="collapsed")
        if img_file: st.image(Image.open(img_file))
    with c2:
        st.markdown(f'<div class="result-box">{st.session_state.res_img}</div>', unsafe_allow_html=True)
    
    if st.button("QUÉT & DỊCH", key="btn2"):
        if img_file:
            res_ocr = reader.readtext(np.array(Image.open(img_file)), detail=0)
            raw_ocr = " ".join(res_ocr)
            st.session_state.res_img = translate_with_llm(raw_ocr, "quét từ ảnh")
            st.rerun()

with tab_doc:
    doc_file = st.file_uploader("Chọn file", type=['pdf','docx','txt'], label_visibility="collapsed")
    if st.button("DỊCH TÀI LIỆU", key="btn3"):
        if doc_file:
            st.session_state.res_doc = translate_with_llm(read_file(doc_file), "tài liệu")
    if st.session_state.res_doc:
        st.markdown(f'<div class="result-box">{st.session_state.res_doc[:2000]}...</div>', unsafe_allow_html=True)