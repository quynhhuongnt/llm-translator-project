import streamlit as st
import google.generativeai as genai
import easyocr
from PIL import Image
import PyPDF2
import docx
import hashlib
from concurrent.futures import ThreadPoolExecutor

# =====================
# CONFIG
# =====================
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
MODEL_NAME = "gemini-1.5-flash"  # tối ưu tốc độ
MAX_CHARS = 2000                 # ~400–500 từ
MAX_WORKERS = 5                  # số luồng song song

# =====================
# INIT
# =====================
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

reader = easyocr.Reader(['en'], gpu=False)
cache = {}

# =====================
# UTILS
# =====================
def hash_text(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def chunk_text(text, max_chars=MAX_CHARS):
    chunks, current = [], ""
    for para in text.split("\n"):
        if len(current) + len(para) < max_chars:
            current += para + "\n"
        else:
            chunks.append(current)
            current = para + "\n"
    if current:
        chunks.append(current)
    return chunks

# =====================
# TRANSLATION
# =====================
def translate_text(text):
    key = hash_text(text)
    if key in cache:
        return cache[key]

    prompt = f"""Translate English to Vietnamese.
Keep meaning accurate and natural.

Text:
{text}
"""
    response = model.generate_content(prompt)
    result = response.text
    cache[key] = result
    return result

def translate_document_fast(text):
    chunks = chunk_text(text)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(translate_text, chunks))

    return "\n".join(results)

# =====================
# OCR
# =====================
def resize_image(img, max_width=1200):
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    return img

@st.cache_data
def ocr_image(img):
    img = resize_image(img)
    result = reader.readtext(img, detail=0)
    return " ".join(result)

# =====================
# DOCUMENT READING
# =====================
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

# =====================
# UI
# =====================
st.set_page_config(page_title="EN → VI Translator (Gemini)", layout="wide")
st.title("🌐 English → Vietnamese Translator (Optimized)")

mode = st.selectbox(
    "Chọn chế độ dịch",
    ["Văn bản", "Hình ảnh", "Tài liệu"]
)

# -------- TEXT --------
if mode == "Văn bản":
    text = st.text_area("Nhập văn bản tiếng Anh", height=250)
    if st.button("🚀 Dịch"):
        if text.strip():
            st.success("Đang dịch...")
            st.write(translate_text(text))

# -------- IMAGE --------
elif mode == "Hình ảnh":
    file = st.file_uploader("Upload ảnh (PNG / JPG)", type=["png", "jpg", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, caption="Ảnh gốc", use_column_width=True)

        if st.button("🚀 OCR + Dịch"):
            with st.spinner("Đang OCR..."):
                text = ocr_image(img)

            st.subheader("📄 Văn bản trích xuất")
            st.write(text)

            st.subheader("🇻🇳 Bản dịch")
            st.write(translate_text(text))

# -------- DOCUMENT --------
elif mode == "Tài liệu":
    file = st.file_uploader("Upload PDF / DOCX (≤ 30 trang)", type=["pdf", "docx"])

    if file:
        if file.name.endswith(".pdf"):
            text = read_pdf(file)
        else:
            text = read_docx(file)

        st.info(f"Số ký tự: {len(text)}")

        if st.button("🚀 Dịch tài liệu"):
            st.warning("Đang dịch (song song, tối ưu tốc độ)...")
            translated = translate_document_fast(text)
            st.subheader("📘 Bản dịch")
            st.write(translated)
