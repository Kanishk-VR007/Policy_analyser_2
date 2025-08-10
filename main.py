from fastapi import FastAPI, UploadFile, File
import uvicorn
import fitz  # PyMuPDF
from docx import Document
from bs4 import BeautifulSoup
import io

# Create FastAPI instance
app = FastAPI()

# Health check route for Railway
@app.get("/")
async def root():
    return {"message": "Policy Analyzer is running on Railway!"}

# File upload and analysis endpoint
@app.post("/analyze")
async def analyze_policy(file: UploadFile = File(...)):
    # Read file
    content = await file.read()

    # Detect file type and extract text
    if file.filename.lower().endswith(".pdf"):
        pdf = fitz.open(stream=content, filetype="pdf")
        text = "\n".join([page.get_text() for page in pdf])

    elif file.filename.lower().endswith(".docx"):
        doc = Document(io.BytesIO(content))  # ✅ Fixed to read from memory
        text = "\n".join([para.text for para in doc.paragraphs])

    elif file.filename.lower().endswith((".html", ".htm")):
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text()

    else:
        return {"error": "Unsupported file format"}

    return {
        "filename": file.filename,
        "text_preview": text[:300]  # Show only first 300 chars
    }

# Entry point for local dev (Railway will ignore this and use Procfile)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
