import os
import json
import pandas as pd
import pdfplumber

MAX_CSV_FILES = 100  # Only load the first 100 CSV files

def load_json_files(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), encoding="utf-8") as f:
                content = json.load(f)
                for item in content:
                    text_blob = "\n".join(f"{k}: {v}" for k, v in item.items())
                    data.append((file, text_blob))
    return data


def load_csv_files(folder_path: str, files: list = None) -> list:
    docs = []
    if files is None:
        files = sorted(os.listdir(folder_path))
    for filename in files:
        if not filename.endswith(".csv"):
            continue
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)
            text_data = df.to_string()
            docs.append((filename, text_data))
        except Exception as e:
            print(f"[CSV ERROR] Could not load {filename}: {e}")
    return docs

def load_pdf_files(folder):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder, file)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                data.append((file, text))
    return data

if __name__ == "__main__":
    json_data = load_json_files("data/jsons") 
    csv_data = load_csv_files("data/csvs")
    pdf_data = load_pdf_files("data/pdfs")

    all_data = json_data + csv_data + pdf_data
    print(f"Loaded {len(all_data)} documents.")
