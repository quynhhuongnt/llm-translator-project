import streamlit as st
from google import genai
import easyocr
import numpy as np
from PIL import Image
import PyPDF2

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="EN-VI Translator (Gemini)", layout="wide")
MODEL_NAME = "models/gemini-1.0-pro"

# ======================
# GEMINI CLIENT (SDK MỚI)
# ======================
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ======================
# OCR
# ======================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en", "vi"], gpu=False)

ocr_reader = load_ocr()

# ======================
# UTILS
# ======================
def split_text(text, max_len=1200):
    words = text.split()
    chunks, cur, length = [], [], 0
    for w in words:
        cur.append(w)
        length += len(w) + 1
        if length >= max_len:
            chunks.append(" ".join(cur))
            cur, length = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ======================
# TRANSLATION (🔥 SIMPLE & STABLE)
# ======================
def translate_chunk(chunk: str) -> str:
    prompt = (
        "Translate the following English text into Vietnamese.\n"
        "Keep meaning accurate, fluent, and natural.\n\n"
        f"{chunk}"
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


def translate_text(text: str) -> str:
    chunks = split_text(text)
    results = []

    for chunk in chunks:   # ❗ KHÔNG DÙNG THREAD
        results.append(translate_chunk(chunk))

    return "\n".join(results)

# ======================
# UI
# ======================
st.title("🇬🇧➡🇻🇳 English – Vietnamese Translator (Gemini 1.0 Pro)")

tab1, tab2, tab3 = st.tabs(["📝 Văn bản", "🖼️ Hình ảnh", "📄 PDF"])

# ---- TEXT ----
with tab1:
    text = st.text_area("Nhập văn bản tiếng Anh", height=250)
    if st.button("Dịch văn bản"):
        if text.strip():
            with st.spinner("Đang dịch..."):
                st.write(translate_text(text))

# ---- IMAGE ----
with tab2:
    img_file = st.file_uploader("Upload ảnh", type=["png", "jpg", "jpeg"])
    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Ảnh gốc", use_column_width=True)

        if st.button("OCR + Dịch"):
            with st.spinner("Đang xử lý..."):
                text = " ".join(
                    ocr_reader.readtext(np.array(image), detail=0)
                )
                st.write(translate_text(text))

# ---- PDF ----
with tab3:
    pdf_file = st.file_uploader("Upload PDF (≤30 trang)", type=["pdf"])
    if pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages[:30]:
            if page.extract_text():
                text += page.extract_text() + "\n"

        if st.button("Dịch PDF"):
            with st.spinner("Đang dịch PDF..."):
                st.write(translate_text(text))
