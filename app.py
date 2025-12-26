import streamlit as st
import google.generativeai as genai
import easyocr
from PIL import Image
import PyPDF2
import docx
from concurrent.futures import ThreadPoolExecutor

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="EN → VI Translator (Gemini 1.0 Pro)",
    layout="wide"
)

st.title("🌐 English → Vietnamese Translator (Gemini 1.0 Pro)")

# =============================
# LOAD SECRET
# =============================
if "GEMINI_API_KEY" not in st.secrets:
    st.error("❌ Missing GEMINI_API_KEY in Streamlit Secrets")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# =============================
# MODEL (STABLE)
# =============================
MODEL_NAME = "models/gemini-1.0-pro"
model = genai.GenerativeModel(MODEL_NAME)

# =============================
# CONSTANTS (SAFE)
# =============================
MAX_CHARS = 1500     # an toàn cho gemini-1.0-pro
MAX_WORKERS = 2      # Streamlit Cloud ổn định

# =============================
# INIT OCR
# =============================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

ocr_reader = load_ocr()

# =============================
# TEXT CHUNKING
# =============================
def chunk_text(text: str):
    chunks, current = [], ""

    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) < MAX_CHARS:
            current += para + "\n"
        else:
            chunks.append(current)
            current = para + "\n"

    if current.strip():
        chunks.append(current)

    return chunks

# =============================
# TRANSLATION (SAFE)
# =============================
def translate_chunk(chunk: str) -> str:
    if not chunk or len(chunk.strip()) < 5:
        return ""

    prompt = (
        "Translate the following English text into Vietnamese.\n"
        "Keep the meaning accurate and natural.\n\n"
        f"Text:\n{chunk}"
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048
            }
        )
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

def translate_text(text: str) -> str:
    chunks = chunk_text(text)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(translate_chunk, chunks):
            results.append(result)

    return "\n".join(results)

# =============================
# OCR
# =============================
def resize_image(img: Image.Image, max_width=1200):
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)))
    return img

@st.cache_data(show_spinner=False)
def ocr_image(img: Image.Image) -> str:
    img = resize_image(img)
    result = ocr_reader.readtext(img, detail=0)
    return " ".join(result)

# =============================
# DOCUMENT READERS
# =============================
def read_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

def read_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())

# =============================
# UI
# =============================
mode = st.selectbox(
    "Chọn chế độ dịch",
    ["Văn bản", "Hình ảnh", "Tài liệu"]
)

# -------- TEXT --------
if mode == "Văn bản":
    text = st.text_area("Nhập văn bản tiếng Anh", height=250)

    if st.button("🚀 Dịch"):
        if len(text.strip()) < 5:
            st.warning("⚠️ Nội dung quá ngắn để dịch")
        else:
            with st.spinner("Đang dịch..."):
                result = translate_text(text)

            st.subheader("🇻🇳 Bản dịch")
            st.write(result)

# -------- IMAGE --------
elif mode == "Hình ảnh":
    file = st.file_uploader(
        "Upload ảnh (PNG / JPG)",
        type=["png", "jpg", "jpeg"]
    )

    if file:
        img = Image.open(file)
        st.image(img, caption="Ảnh gốc", use_column_width=True)

        if st.button("🚀 OCR + Dịch"):
            with st.spinner("OCR ảnh..."):
                extracted = ocr_image(img)

            if not extracted.strip():
                st.warning("⚠️ Không nhận diện được văn bản")
            else:
                st.subheader("📄 Văn bản trích xuất")
                st.write(extracted)

                with st.spinner("Dịch sang tiếng Việt..."):
                    translated = translate_text(extracted)

                st.subheader("🇻🇳 Bản dịch")
                st.write(translated)

# -------- DOCUMENT --------
elif mode == "Tài liệu":
    file = st.file_uploader(
        "Upload PDF / DOCX (≤ 30 trang)",
        type=["pdf", "docx"]
    )

    if file:
        if file.name.endswith(".pdf"):
            text = read_pdf(file)
        else:
            text = read_docx(file)

        if len(text.strip()) < 50:
            st.warning("⚠️ Không đọc được nội dung tài liệu")
        else:
            st.info(f"Số ký tự: {len(text)}")

            if st.button("🚀 Dịch tài liệu"):
                with st.spinner("Đang dịch (ổn định, an toàn)..."):
                    translated = translate_text(text)

                st.subheader("📘 Bản dịch")
                st.write(translated)
