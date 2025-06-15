import os
import pandas as pd
from PIL import Image
import pytesseract
import pdfplumber
from docx import Document

# Internal utility
from modules.data_utils import normalize_columns

api_key = os.environ.get("TOGETHER_API_KEY")

def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower().strip('.')

    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            return '\n'.join([page.extract_text() or '' for page in pdf.pages]), None

    elif ext == 'txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read(), None
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                return f.read(), None

    elif ext in ['doc', 'docx']:
        return extract_text_from_docx(file_path), None

    elif ext in ['csv', 'xlsx']:
        try:
            if ext == 'csv':
                try:
                    df = pd.read_csv(file_path)
                except UnicodeDecodeError:
                    print("⚠️ UTF-8 failed, trying ISO-8859-1...")
                    df = pd.read_csv(file_path, encoding='ISO-8859-1')
            else:
                df = pd.read_excel(file_path)

            df, col_map = normalize_columns(df)
            return df, col_map

        except Exception as e:
            print("❌ Error reading tabular file:", e)
            raise e

    elif ext in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
        return extract_text_from_image(file_path), None

    else:
        print(f"🚨 DEBUG: File extension = .{ext}")
        raise ValueError(f"Unsupported file type: .{ext}")



def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"❌ Failed to extract text from image: {e}"


def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"❌ Failed to read .docx file: {e}")
        return None