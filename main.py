from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import os
from model import rag_chain, Loader, RecursiveCharacterTextSplitter, Chroma, HuggingFaceEmbeddings

app = FastAPI()

UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

retriever = None

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Renders the home page with buttons for uploading a PDF and running the model.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file and set up the retriever.
    """
    global retriever
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(await file.read())

    # Process the uploaded file
    loader = Loader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    # Create retriever
    embedding = HuggingFaceEmbeddings()
    vector_db = Chroma.from_documents(documents=docs, embedding=embedding)
    retriever = vector_db.as_retriever()

    return {"message": "File uploaded and retriever set successfully!", "file_name": file.filename}

@app.post("/query")
async def query_model(question: str = Form(...)):
    """
    Endpoint to query the model.
    """
    global retriever
    if not retriever:
        return {"error": "No PDF uploaded. Please upload a PDF first."}

    # try:
        # Use the retriever in the RAG chain
    # rag_chain["context"] = retriever
    output = rag_chain.invoke(question)
    answer = output.split("Answer: ")[-1].strip() if "Answer: " in output else output.strip()
    return {"question": question, "answer": answer}
    # except Exception as e:
    #     return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
