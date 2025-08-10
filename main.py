from fastapi import FastAPI, UploadFile, File
import uvicorn
import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup
import openai
import faiss
from sentence_transformers import SentenceTransformer

# Create FastAPI instance
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Policy Analyzer is running!"}

@app.post("/analyze")
async def analyze_policy(file: UploadFile = File(...)):
    # Read file
    content = await file.read()

    # Detect file type and extract text
    if file.filename.lower().endswith(".pdf"):
        pdf = fitz.open(stream=content, filetype="pdf")
        text = "\n".join([page.get_text() for page in pdf])
    elif file.filename.lower().endswith(".docx"):
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif file.filename.lower().endswith(".html") or file.filename.lower().endswith(".htm"):
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text()
    else:
        return {"error": "Unsupported file format"}

    # TODO: Replace with your actual policy analysis logic
    return {"filename": file.filename, "text_preview": text[:300]}

if __name__ == "__main__":
    # Local run only — Railway will use Procfile
    uvicorn.run(app, host="0.0.0.0", port=8000)


