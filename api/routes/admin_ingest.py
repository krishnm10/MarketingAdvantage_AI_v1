# ==========================================
# 🧠 admin_ingest.py — Manage RAG Knowledge Base (Content-as-a-Service)
# ==========================================
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from api.services.rag_service import ingest_content, clear_vector_store, retrieve_relevant_chunks
import os
import traceback
from api.services.manual_ingest import ingest_file_to_vector_and_db

router = APIRouter(prefix="/admin", tags=["Admin / RAG Management"])


# ==============================================================
# 🔹 POST /admin/ingest_text — Ingest raw text into RAG DB
# ==============================================================
@router.post("/ingest_text")
async def ingest_text_content(
    doc_id: str = Form(...),
    text: str = Form(...),
    category: str = Form("general"),
    source: str = Form("manual_upload")
):
    """
    Ingest a text document into the RAG vector database.
    Useful for adding blogs, FAQs, or brand copy.

    Example curl:
    curl -X POST http://localhost:8000/admin/ingest_text \
      -F "doc_id=blog_2025_trends" \
      -F "text=AI is transforming marketing content..." \
      -F "category=marketing" \
      -F "source=blog"
    """
    try:
        ingest_content(
            doc_id=doc_id,
            text=text,
            metadata={"category": category, "source": source}
        )
        return {
            "status": "success",
            "message": f"Document '{doc_id}' successfully added to RAG DB.",
            "metadata": {"category": category, "source": source}
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to ingest content: {str(e)}")


# ==============================================================
# 🔹 POST /admin/ingest_file — Ingest a text or PDF file into RAG DB
# ==============================================================
@router.post("/ingest_file")
async def ingest_file_content(file: UploadFile = File(...)):
    """
    Upload and ingest a .txt or .pdf file into the RAG vector database.
    Extracts and embeds file text automatically.

    Example curl:
    curl -X POST -F "file=@example.txt" http://localhost:8000/admin/ingest_file
    """
    try:
        filename = file.filename
        file_ext = os.path.splitext(filename)[-1].lower()
        content_text = ""

        if file_ext == ".txt":
            content_text = (await file.read()).decode("utf-8", errors="ignore")
        elif file_ext == ".pdf":
            import fitz  # PyMuPDF for PDF text extraction
            pdf_bytes = await file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            content_text = "\n".join(page.get_text("text") for page in doc)
        else:
            raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

        if not content_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in file.")

        ingest_content(
            doc_id=os.path.splitext(filename)[0],
            text=content_text,
            metadata={"source": "file_upload", "filename": filename}
        )

        return {"status": "success", "message": f"File '{filename}' ingested successfully."}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")


# ==============================================================
# 🔹 GET /admin/search_rag — Retrieve sample matches from RAG
# ==============================================================
@router.get("/search_rag")
def search_rag(query: str):
    """
    Test search: Retrieves top relevant chunks from RAG based on a query.
    Useful for debugging or verifying embeddings.
    """
    try:
        chunks = retrieve_relevant_chunks(query)
        return {
            "query": query,
            "matches_found": len(chunks),
            "results": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ==============================================================
# 🔹 DELETE /admin/clear_rag — Clear the entire RAG database
# ==============================================================
@router.delete("/clear_rag")
def clear_rag():
    """
    Clears the entire vector database (removes all stored embeddings).
    ⚠️ Use this carefully — cannot be undone.
    """
    try:
        clear_vector_store()
        return {"status": "cleared", "message": "RAG vector store has been reset successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear RAG DB: {str(e)}")

# ==============================================================
# 🔹 Manual TXT/PDF insert in to RAG database
# ==============================================================

@router.post("/ingest_file")
async def ingest_file_content(file: UploadFile = File(...)):
    """Upload and ingest a PDF or TXT file into RAG + Postgres."""
    try:
        filename = file.filename
        upload_path = f"./data/uploads/{filename}"
        os.makedirs("./data/uploads", exist_ok=True)

        with open(upload_path, "wb") as f:
            f.write(await file.read())

        ingest_file_to_vector_and_db(upload_path, source="manual_upload")

        return {"status": "success", "message": f"File '{filename}' ingested and categorized."}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {str(e)}")