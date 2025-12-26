import streamlit as st
from google import genai
import easyocr
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
import io
import os

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="EN-VI Translator (Gemini)", layout="wide")

MODEL_NAME = "models/gemini-1.0-pro"

# =====================
# API KEY
# =====================
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# =====================
# OCR
# =====================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en", "vi"], gpu=False)

reader = load_ocr()

# =====================
# UTILS
# =====================
def split_text(text, max_len=1500):
    words = text.split()
    chunks, current = [], []
    length = 0
    for w in words:
        length += len(w) + 1
        current.append(w)
        if length >= max_len:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


# =====================
# TRANSLATION (🔥 NO temperature, NO config)
# =====================
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

    with ThreadPoolExecutor(max_workers=3) as executor:
        for r in executor.map(translate_chunk, chunks):
            results.append(r)

    return "\n".join(results)

# =====================
# UI
# =====================
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
            with st.spinner("Đang nhận dạng & dịch..."):
                text = " ".join(reader.readtext(np.array(image), detail=0))
                st.write(translate_text(text))

# ---- PDF ----
with tab3:
    pdf_file = st.file_uploader("Upload PDF (≤30 trang)", type=["pdf"])
    if pdf_file:
        reader_pdf = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader_pdf.pages[:30]:
            text += page.extract_text() + "\n"

        if st.button("Dịch PDF"):
            with st.spinner("Đang dịch PDF..."):
                st.write(translate_text(text))
