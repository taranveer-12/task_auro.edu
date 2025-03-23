from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
import logging
import PyPDF2 
import docx  

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql://postgres:Admin123*@localhost/document_db"  

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def generate_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def extract_text_from_file(file: UploadFile) -> str:
    content = file.file.read()
    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file.file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    elif file.filename.endswith(".txt"):
        return content.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Supported types: PDF, DOCX, TXT.")

class Question(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None

@app.post("/ingest/")
async def ingest_document(file: UploadFile = File(...)):
    try:
        text = extract_text_from_file(file)
        logger.info(f"Extracted text from file: {file.filename}")

        embedding = generate_embedding(text)
        embedding_list = embedding.tolist()
        logger.info(f"Generated embedding for file: {file.filename}")

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO docs (id, content, embedding) VALUES (%s, %s, %s)",
            (file.filename, text, embedding_list)  
        )
        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"Document {file.filename} ingested successfully.")
        return {"message": "Document ingested successfully", "filename": file.filename}
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_answer_with_huggingface(context: str, question: str) -> str:
    try:
        result = qa_pipeline(question=question, context=context)
        return result["answer"]
    except Exception as e:
        logger.error(f"Error generating answer with Hugging Face: {e}")
        return "Sorry, I cannot provide an answer at the moment. Please try again later."
    
@app.post("/select-documents/")
async def select_documents(selection: DocumentSelection):
    """
    Select which documents should be included in the Q&A process.
    """
    try:
        conn = await get_db_connection()
        await conn.execute(
            "UPDATE docs SET included_in_qa = FALSE"
        )
        await conn.execute(
            "UPDATE docs SET included_in_qa = TRUE WHERE id = ANY($1)",
            selection.document_ids
        )
        await conn.close()

        logger.info(f"Selected documents for Q&A: {selection.document_ids}")
        return {"message": "Documents selected successfully"}
    except Exception as e:
        logger.error(f"Error selecting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_answer_with_rag(context: str, question: str) -> str:
    """
    Generate an answer using OpenAI's GPT-3 (RAG).
    """
    try:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use GPT-3 for RAG
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logger.error(f"Error generating answer with RAG: {e}")
        return "Sorry, I cannot provide an answer at the moment. Please try again later."


@app.post("/ask/")
async def ask_question(question: Question):
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if question.document_ids:
            cur.execute(
                "SELECT id, content, embedding FROM docs WHERE id = ANY(%s)",
                (question.document_ids,)
            )
        else:
            cur.execute("SELECT id, content, embedding FROM docs")

        documents = cur.fetchall()
        cur.close()
        conn.close()

        if not documents:
            raise HTTPException(status_code=404, detail="No documents found")

        question_embedding = generate_embedding(question.question).flatten()
        logger.info(f"Generated embedding for question: {question.question}")

        similarities = []
        for doc in documents:
            doc_embedding = np.array(doc[2]).flatten()
            similarity = np.dot(question_embedding, doc_embedding.T) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc[0], doc[1], similarity))

        similarities.sort(key=lambda x: x[2], reverse=True)
        most_relevant_doc = similarities[0][1]
        logger.info(f"Most relevant document: {similarities[0][0]}")

        answer = generate_answer_with_huggingface(most_relevant_doc, question.question)
        logger.info(f"Generated answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)