import os
import pandas as pd
from pdf2image import convert_from_path
import pytesseract

# Paths
PDF_FOLDER = "./data/Greenberg"
OUTPUT_PARQUET = "data/processed/greenberg_processed.parquet"

# Store results here
data = []

# Loop over PDFs
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        print(f"OCR'ing: {filename}")

        try:
            pages = convert_from_path(pdf_path)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            continue

        text = ""
        for i, page in enumerate(pages):
            page_text = pytesseract.image_to_string(page)
            text += f"\n\n{page_text}"
            print(page_text)

        data.append({
            "filename": filename,
            "text": text
        })

# Convert to DataFrame and save as Parquet
df = pd.DataFrame(data)
df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", index=False)

print(f"\n✅ Saved all OCR data to: {OUTPUT_PARQUET}")

