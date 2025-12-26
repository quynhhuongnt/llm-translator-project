import streamlit as st
from google import genai
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

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# =============================
# MODEL CONFIG
# =============================
MODEL_NAME = "models/gemini-1.0-pro"

# =============================
# CONSTANTS
# =============================
MAX_CHARS = 1500
MAX_WORKERS = 2   # Cloud-safe

# =============================
# OCR INIT
# =============================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

ocr_reader = load_ocr()

# =============================
# CHUNKING
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
# TRANSLATION
# =============================
def translate_chunk(chunk: str) -> str:
    if len(chunk.strip()) < 5:
        return ""

    prompt = (
        "Translate the following English text into Vietnamese.\n"
        "Keep the meaning accurate and natural.\n\n"
        f"Text:\n{chunk}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            temperature=0.2,
            max_output_tokens=2048
        )
        return response.text
    except Exception as e:
        return f"❌ Error: {e}"

def translate_text(text: str) -> str:
    chunks = chunk_text(text)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for r in executor.map(translate_chunk, chunks):
            results.append(r)

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
            st.warning("⚠️ Nội dung quá ngắn")
        else:
            with st.spinner("Đang dịch..."):
                result = translate_text(text)
            st.subheader("🇻🇳 Bản dịch")
            st.write(result)

# -------- IMAGE --------
elif mode == "Hình ảnh":
    file = st.file_uploader("Upload ảnh", type=["png", "jpg", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, use_column_width=True)

        if st.button("🚀 OCR + Dịch"):
            with st.spinner("OCR ảnh..."):
                extracted = ocr_image(img)

            if not extracted.strip():
                st.warning("⚠️ Không nhận diện được văn bản")
            else:
                st.subheader("📄 Văn bản trích xuất")
                st.write(extracted)

                with st.spinner("Dịch..."):
                    translated = translate_text(extracted)

                st.subheader("🇻🇳 Bản dịch")
                st.write(translated)

# -------- DOCUMENT --------
elif mode == "Tài liệu":
    file = st.file_uploader("Upload PDF / DOCX", type=["pdf", "docx"])

    if file:
        text = read_pdf(file) if file.name.endswith(".pdf") else read_docx(file)

        if len(text.strip()) < 50:
            st.warning("⚠️ Không đọc được nội dung")
        else:
            st.info(f"Số ký tự: {len(text)}")

            if st.button("🚀 Dịch tài liệu"):
                with st.spinner("Đang dịch..."):
                    translated = translate_text(text)

                st.subheader("📘 Bản dịch")
                st.write(translated)

