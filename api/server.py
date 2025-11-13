import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    try:
        from scripts.answer_synthesis import synthesize_answer

        result = synthesize_answer(query.question, debug=True)

        return {
            "query": query.question,
            "answer": result.get("answer", "No answer"),
            "sources": [f"{c['doc_id']}::{c['chunk_id']}" for c in result.get("retrieved_chunks", [])]
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Exception in /ask endpoint:\n{tb}")  # Logs detailed traceback in server console
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use import string for the app, not the app object, to enable reload
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
