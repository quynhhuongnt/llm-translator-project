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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Khởi tạo Gemini Client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    MODEL_NAME = "models/gemini-2.0-flash-lite"
except Exception as e:
    st.error(f"Lỗi khởi tạo API: {e}")

# =========================================================
# 2. HÀM XỬ LÝ LOGIC
# =========================================================

def translate_engine(contents, temperature=0.2, max_tokens=2048):
    """Gọi Gemini API không qua LangSmith"""
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
    """Tạo file PDF hỗ trợ Tiếng Việt"""
    pdf = FPDF()
    pdf.add_page()
    
    # KIỂM TRA FONT: Nếu bạn có file font .ttf, hãy bỏ comment 2 dòng dưới
    # pdf.add_font('VietnameseFont', '', 'Arial.ttf', uni=True)
    # pdf.set_font('VietnameseFont', size=12)
    
    # Tạm thời dùng font mặc định (Lưu ý: Font mặc định có thể lỗi dấu nếu không add font .ttf)
    pdf.set_font("Helvetica", size=12)
    
    # Xử lý nội dung văn bản
    pdf.multi_cell(0, 10, txt=text)
    return pdf.output()

# =========================================================
# 3. GIAO DIỆN TÙY CHỈNH (CSS & SIDEBAR)
# =========================================================
st.set_page_config(page_title="Deep Learning Translator", layout="wide", page_icon="🇬🇧🇻🇳")

with st.sidebar:
    st.header("⚙️ Cấu hình Model")
    temp_val = st.slider("Temperature (Độ sáng tạo)", 0.0, 1.0, 0.2, 0.1)
    max_token_val = st.number_input("Max Output Tokens", 100, 8192, 2048)
    
    st.divider()
    st.markdown("### 📝 Tùy chỉnh Prompt")
    system_prompt = st.text_area(
        "Yêu cầu dịch thuật:", 
        value="Bạn là một biên dịch viên chuyên nghiệp. Hãy dịch nội dung sau sang tiếng Việt một cách tự nhiên.",
    )

st.markdown("""
<style>
    .stTextArea textarea { font-size: 16px; height: 300px; }
    .stButton button { background-color: #1a73e8; color: white; font-weight: bold; width: 100%; border-radius: 8px; }
    .result-box { border: 1px solid #d3d3d3; border-radius: 0.5rem; padding: 1rem; height: 300px; background-color: #f0f2f6; overflow-y: auto; white-space: pre-wrap; font-size: 16px; }
    .lang-header { font-weight: bold; font-size: 18px; color: #1a73e8; margin-bottom: 10px; display: block; }
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
        text_input = st.text_area("Input", placeholder="Nhập văn bản ...", label_visibility="collapsed", key="txt_in")
    with col2:
        st.markdown('<span class="lang-header"> KẾT QUẢ DỊCH </span>', unsafe_allow_html=True)
        res_txt = st.empty()
        res_txt.markdown('<div class="result-box">Đang đợi nội dung...</div>', unsafe_allow_html=True)

    if st.button("🚀 BẮT ĐẦU DỊCH", key="btn_text"):
        if text_input.strip():
            with st.spinner("Đang xử lý..."):
                ans = translate_engine(f"{system_prompt}\n\n{text_input}", temp_val, max_token_val)
                res_txt.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)
        else:
            st.warning("Vui lòng nhập văn bản!")

# --- TAB 2: DỊCH HÌNH ẢNH --- (Tương tự nhưng không có nút tải PDF vì ảnh thường dịch ngắn)
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
            with st.spinner("Đang phân tích ảnh..."):
                img_data = Image.open(up_img)
                ans = translate_engine([system_prompt, img_data], temp_val, max_token_val)
                res_img.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)

# --- TAB 3: DỊCH TÀI LIỆU & TẢI PDF ---
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
            with st.spinner("Đang phân tích tài liệu..."):
                bytes_data = up_doc.read()
                content_payload = [types.Part.from_bytes(data=bytes_data, mime_type=up_doc.type), system_prompt]
                ans = translate_engine(content_payload, temp_val, max_token_val)
                res_doc.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)
                
                # Nút tải PDF
                pdf_data = export_to_pdf(ans)
                st.download_button(
                    label="📥 Tải xuống bản dịch (.pdf)",
                    data=pdf_data,
                    file_name=f"translated_{up_doc.name}.pdf",
                    mime="application/pdf"
                )
