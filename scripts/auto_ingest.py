import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
from pathlib import Path

INGEST_DIR = "data/new_uploads"

class IngestHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = event.src_path
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            print(f"\n [{datetime.now().strftime('%H:%M:%S')}] New PDF detected: {path}")
            try:
                print("→ PDF → Markdown pipeline...")
                subprocess.run(["python", "scripts/pdf_to_markdown.py", path], check=True)

                print("→ Running semantic chunker...")
                subprocess.run(["python", "scripts/semantic_chunker.py"], check=True)

                print("→ Updating embeddings in Postgres...")
                subprocess.run(["python", "scripts/embed_chunks.py"], check=True)

                print("→ Rebuilding BM25 index...")
                subprocess.run(["python", "scripts/build_bm25_index.py"], check=True)

                print(f" Auto-ingestion complete for {path}\n")
            except subprocess.CalledProcessError as e:
                print(f" Auto-ingestion failed: {e}")

def start_ingest_watcher():
    Path(INGEST_DIR).mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(IngestHandler(), INGEST_DIR, recursive=False)
    observer.start()
    print(f" Watching {INGEST_DIR} for new PDFs")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    print("\n=== AUTO INGESTION PIPELINE STARTED ===\n")
    start_ingest_watcher()





    
