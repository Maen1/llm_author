import os
from pathlib import Path
import logging
import pandas as pd
from pdfminer.high_level import extract_text
import pytesseract
from pdf2image import convert_from_path

logging.basicConfig(level=logging.INFO)
raw_data="data/Krauss"
processed_data = "data/processed/"
class DataCollector:
    def __init__(self, input_dir: str = raw_data):
        self.input_dir = Path(input_dir)
        
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        try:
            return extract_text(pdf_path)
        except Exception as e:
            logging.warning(f"PDF extraction failed for {pdf_path}, trying OCR: {e}")
            images = convert_from_path(pdf_path)
            return " ".join(pytesseract.image_to_string(img) for img in images)

    def run(self) -> pd.DataFrame:
        corpus = []
        for file in self.input_dir.glob("*"):
            print(file)
            if file.suffix == ".pdf":
                text = self._extract_text_from_pdf(file)
                print(text)
            elif file.suffix in (".txt", ".md"):
                text = file.read_text()
            else:
                continue
            corpus.append({"source": file.name, "text": text})
        
        df = pd.DataFrame(corpus)
        df.to_parquet(processed_data + "krauss.parquet")  # Efficient storage
        return df

if __name__ == "__main__":
    collector = DataCollector()
    corpus_df = collector.run()
