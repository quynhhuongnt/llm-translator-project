import streamlit as st
from PIL import Image
from google import genai
from dotenv import load_dotenv
from google.genai import types
import os
import io
from fpdf import FPDF

# =========================================================
# 1. CẤU HÌNH HỆ THỐNG & API KEYS
# =========================================================
load_dotenv() 

# Lấy Key từ Streamlit Secrets (Cloud) hoặc .env (Local)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

try:
    if not GEMINI_API_KEY:
        st.error("❌ Thiếu GEMINI_API_KEY! Hãy thêm vào Secrets trên Streamlit Cloud.")
        st.stop()
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    MODEL_NAME = "models/gemini-2.0-flash-lite" # Model tốc độ cao, đa phương thức
except Exception as e:
    st.error(f"Lỗi khởi tạo API: {e}")

# =========================================================
# 2. HÀM XỬ LÝ LOGIC
# =========================================================

def translate_engine(contents, temperature=0.2, max_tokens=2048):
    """Gửi yêu cầu đến Gemini API"""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
            )
        )
        return response.text
    except Exception as e:
        if "429" in str(e):
            return "⚠️ LỖI QUOTA (429): Bạn đã hết lượt dùng miễn phí. Vui lòng đợi 60s."
        return f"❌ LỖI HỆ THỐNG: {str(e)}"

def export_to_pdf(text):
    """Xuất văn bản ra PDF hỗ trợ Tiếng Việt (Yêu cầu file font .ttf trong thư mục)"""
    pdf = FPDF()
    pdf.add_page()
    
    # Kiểm tra file font để tránh crash nếu bạn quên upload Arial.ttf
    font_filename = "Arial.ttf" # Bạn có thể đổi thành Roboto-Regular.ttf tùy file bạn có
    if os.path.exists(font_filename):
        pdf.add_font("VietnameseFont", "", font_filename)
        pdf.set_font("VietnameseFont", size=12)
    else:
        pdf.set_font("Helvetica", size=12) # Font dự phòng (sẽ lỗi dấu tiếng Việt)
        st.warning(f"⚠️ Không tìm thấy file {font_filename}. PDF sẽ bị lỗi font tiếng Việt.")

    # Chia văn bản thành nhiều dòng
    pdf.multi_cell(0, 10, txt=text)
    return pdf.output()

# =========================================================
# 3. GIAO DIỆN TÙY CHỈNH (CSS & SIDEBAR)
# =========================================================
st.set_page_config(page_title="Deep Learning Translator", layout="wide", page_icon="🇬🇧🇻🇳")

with st.sidebar:
    st.header("⚙️ Cấu hình Model")
    st.info("Tinh chỉnh tham số giúp kiểm soát độ chính xác.")
    
    temp_val = st.slider("Temperature (Độ sáng tạo)", 0.0, 1.0, 0.2, 0.1)
    max_token_val = st.number_input("Max Output Tokens", 100, 8192, 2048)
    
    st.divider()
    st.markdown("### 📝 Tùy chỉnh Prompt")
    system_prompt = st.text_area(
        "Yêu cầu dịch thuật:", 
        value="Bạn là một biên dịch viên chuyên nghiệp. Hãy dịch nội dung sau sang tiếng Việt một cách tự nhiên, giữ nguyên định dạng Markdown nếu có.",
        help="Ví dụ: Dịch chuyên ngành y khoa, dịch thơ, dịch trang trọng..."
    )

st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px; height: 300px; }
    .stButton button { 
        background-color: #1a73e8; color: white; font-size: 16px; 
        border-radius: 8px; padding: 0.5rem 1rem; width: 100%; font-weight: bold;
    }
    .result-box { 
        border: 1px solid #d3d3d3; border-radius: 0.5rem; padding: 1rem; height: 300px;              
        background-color: #f0f2f6; color: #31333F; overflow-y: auto;
        white-space: pre-wrap; font-size: 16px;
    }
    .lang-header { font-weight: bold; font-size: 18px; margin-bottom: 10px; display: block; color: #1a73e8; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 4. GIAO DIỆN ỨNG DỤNG (UI)
# =========================================================
st.title(" ỨNG DỤNG DỊCH ĐA PHƯƠNG THỨC ")
st.caption("Môn: Kĩ thuật học sâu | SV: Ngô Thị Quỳnh Hương | Tech: Gemini 2.0 Flash Lite")

tab_text, tab_image, tab_doc = st.tabs(["🔤 Văn Bản", "📸 Hình Ảnh", "📂 Tài Liệu"])

# --- TAB 1: DỊCH VĂN BẢN ---
with tab_text:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="lang-header">VĂN BẢN CẦN DỊCH</span>', unsafe_allow_html=True)
        text_input = st.text_area("Input", placeholder="Nhập văn bản tiếng Anh...", label_visibility="collapsed", key="txt_in")
    with col2:
        st.markdown('<span class="lang-header"> VĂN BẢN ĐÃ DỊCH </span>', unsafe_allow_html=True)
        res_txt = st.empty()
        res_txt.markdown('<div class="result-box">Kết quả dịch...</div>', unsafe_allow_html=True)

    if st.button("🚀 BẮT ĐẦU DỊCH", key="btn_text"):
        if text_input.strip():
            with st.spinner("Đang dịch..."):
                full_prompt = f"{system_prompt}\n\nNội dung cần dịch:\n{text_input}"
                ans = translate_engine(full_prompt, temperature=temp_val, max_tokens=max_token_val)
                res_txt.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)
        else:
            st.warning("Vui lòng nhập văn bản!")

# --- TAB 2: DỊCH HÌNH ẢNH ---
with tab_image:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="lang-header">TẢI ẢNH LÊN</span>', unsafe_allow_html=True)
        up_img = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        if up_img: st.image(Image.open(up_img), use_container_width=True)
    with col2:
        st.markdown('<span class="lang-header">KẾT QUẢ TRÍCH XUẤT & DỊCH</span>', unsafe_allow_html=True)
        res_img = st.empty()
        res_img.markdown('<div class="result-box">Đang đợi ảnh...</div>', unsafe_allow_html=True)

    if st.button("🔍 QUÉT & DỊCH ẢNH", key="btn_img"):
        if up_img:
            with st.spinner("Đang phân tích Vision..."):
                img_data = Image.open(up_img)
                content_payload = [system_prompt, img_data]
                ans = translate_engine(content_payload, temperature=temp_val, max_tokens=max_token_val)
                res_img.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)

# --- TAB 3: DỊCH TÀI LIỆU ---
with tab_doc:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<span class="lang-header">TẢI FILE (PDF/TXT)</span>', unsafe_allow_html=True)
        up_doc = st.file_uploader("Upload Doc", type=['pdf', 'txt'], label_visibility="collapsed")
    with col2:
        st.markdown('<span class="lang-header">NỘI DUNG DỊCH</span>', unsafe_allow_html=True)
        res_doc = st.empty()
        res_doc.markdown('<div class="result-box">Đang đợi tài liệu...</div>', unsafe_allow_html=True)

    if st.button("📄 DỊCH TOÀN BỘ FILE", key="btn_doc"):
        if up_doc:
            with st.spinner("Đang đọc tài liệu..."):
                bytes_data = up_doc.read()
                content_payload = [
                    types.Part.from_bytes(data=bytes_data, mime_type=up_doc.type),
                    system_prompt
                ]
                ans = translate_engine(content_payload, temperature=temp_val, max_tokens=max_token_val)
                res_doc.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)
                
                # Tạo file PDF để tải về
                pdf_output = export_to_pdf(ans)
                st.download_button(
                    label="📥 Tải xuống bản dịch (.pdf)", 
                    data=pdf_output, 
                    file_name=f"translated_{up_doc.name}.pdf",
                    mime="application/pdf"
                )
