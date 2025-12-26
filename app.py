import streamlit as st
import google.generativeai as genai
import easyocr
from PIL import Image
import numpy as np
import docx
import PyPDF2
from io import StringIO

# 1. CẤU HÌNH TRANG & GIAO DIỆN CSS
st.set_page_config(page_title="Hương Ngô - AI Translator", layout="wide", page_icon="🌐")

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
        height: 300px; background-color: #f8f9fa; color: #31333F;           
        overflow-y: auto; font-family: sans-serif; font-size: 16px;          
        white-space: pre-wrap;
    }
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .lang-header { font-weight: bold; font-size: 18px; margin-bottom: 10px; display: block; color: #1a73e8; }
</style>
""", unsafe_allow_html=True)

# 2. KHỞI TẠO CÔNG CỤ (BACKEND)
@st.cache_resource
def load_ocr():
    # Tắt GPU vì Streamlit Cloud chạy trên CPU
    return easyocr.Reader(['en'], gpu=False)

def get_model():
    try:
        # Đảm bảo bạn đã thêm GEMINI_API_KEY vào mục Settings > Secrets trên Streamlit
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        # Khởi tạo model với định danh chuẩn để tránh lỗi 404
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Lỗi cấu hình API Key: {e}")
        return None

model = get_model()
reader = load_ocr()

# 3. HÀM XỬ LÝ DỊCH THUẬT (SỬ DỤNG STREAMING)
def translate_stream(text, context="văn bản"):
    if not text.strip(): return
    prompt = f"Bạn là một chuyên gia dịch thuật. Hãy dịch đoạn {context} sau sang tiếng Việt một cách tự nhiên và lưu loát nhất: \n\n{text}"
    try:
        # Kích hoạt stream=True để nhận kết quả từng phần, tối ưu tốc độ phản hồi
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"⚠️ Lỗi kết nối API: {str(e)}"

def read_file_content(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "text/plain":
            text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
        elif uploaded_file.type == "application/pdf":
            pdf = PyPDF2.PdfReader(uploaded_file)
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted: text += extracted + "\n"
        elif "word" in uploaded_file.type or "officedocument" in uploaded_file.type:
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Lỗi đọc tài liệu: {e}")
    return text

# 4. GIAO DIỆN NGƯỜI DÙNG (FRONTEND)
st.title("🚀 ỨNG DỤNG DỊCH ANH - VIỆT CLOUD LLM")
st.markdown("**Sinh viên thực hiện:** Ngô Thị Quỳnh Hương | **Mã SV:** 99048")
st.divider()

tab1, tab2, tab3 = st.tabs(["🔤 Văn Bản", "📸 Hình Ảnh", "📂 Tài Liệu"])

# Sử dụng Session State để duy trì kết quả
if 'res_text' not in st.session_state: st.session_state.res_text = ""

# TAB 1: DỊCH VĂN BẢN
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="lang-header">TIẾNG ANH</span>', unsafe_allow_html=True)
        t_input = st.text_area("Input", height=300, label_visibility="collapsed", key="txt_in")
    with c2:
        st.markdown('<span class="lang-header">TIẾNG VIỆT (AI STREAMING)</span>', unsafe_allow_html=True)
        res_placeholder = st.empty()
        res_placeholder.markdown(f'<div class="result-box">{st.session_state.res_text}</div>', unsafe_allow_html=True)

    if st.button("DỊCH NGAY", key="btn_text"):
        if t_input:
            st.session_state.res_text = ""
            full_res = ""
            # Hiển thị kết quả kiểu gõ chữ (Streaming)
            for chunk in translate_stream(t_input):
                full_res += chunk
                res_placeholder.markdown(f'<div class="result-box">{full_res}</div>', unsafe_allow_html=True)
            st.session_state.res_text = full_res

# TAB 2: DỊCH HÌNH ẢNH (OCR + LLM)
with tab2:
    col_img, col_res = st.columns(2)
    with col_img:
        st.markdown('<span class="lang-header">TẢI ẢNH LÊN</span>', unsafe_allow_html=True)
        img_file = st.file_uploader("Upload", type=['png','jpg','jpeg'], label_visibility="collapsed")
        if img_file: st.image(img_file, use_container_width=True)
    
    with col_res:
        st.markdown('<span class="lang-header">KẾT QUẢ QUÉT & DỊCH</span>', unsafe_allow_html=True)
        res_img_place = st.empty()
        res_img_place.markdown('<div class="result-box"></div>', unsafe_allow_html=True)

    if st.button("BẮT ĐẦU QUÉT & DỊCH", key="btn_img"):
        if img_file:
            with st.spinner("Đang nhận diện chữ..."):
                img_np = np.array(Image.open(img_file))
                ocr_result = reader.readtext(img_np, detail=0)
                ocr_text = " ".join(ocr_result)
            
            prefix = f"**Nội dung nhận diện:** {ocr_text}\n\n**Bản dịch:**\n"
            temp_res = ""
            for chunk in translate_stream(ocr_text, "từ hình ảnh"):
                temp_res += chunk
                res_img_place.markdown(f'<div class="result-box">{prefix + temp_res}</div>', unsafe_allow_html=True)

# TAB 3: DỊCH TÀI LIỆU (PDF/DOCX/TXT)
with tab3:
    doc_file = st.file_uploader("Chọn file tài liệu", type=['pdf','docx','txt'])
    if st.button("DỊCH TOÀN BỘ FILE", key="btn_doc"):
        if doc_file:
            with st.spinner("Đang xử lý tài liệu..."):
                file_content = read_file_content(doc_file)
                res_doc_place = st.empty()
                full_doc_res = ""
                for chunk in translate_stream(file_content, "tài liệu"):
                    full_doc_res += chunk
                    res_doc_place.markdown(f'<div class="result-box">{full_doc_res}</div>', unsafe_allow_html=True)
