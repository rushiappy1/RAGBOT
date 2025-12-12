import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from datetime import datetime
from pathlib import Path

WATCH_DIRS = [
    "data/markdown",
    "data/chunks"
]

class ChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        path = event.src_path
        ext = Path(path).suffix.lower()

        if ext in [".md", ".pdf", ".jsonl"]:
            print(f"\n🧩 [{datetime.now().strftime('%H:%M:%S')}] Change detected: {path}")

            try:
                if ext == ".pdf":
                    print("→ Running PDF → Markdown pipeline...")
                    subprocess.run(["python", "scripts/pdf_to_markdown.py", path], check=True)

                print("→ Running semantic chunker...")
                subprocess.run(["python", "scripts/semantic_chunker.py"], check=True)

                print("→ Updating embeddings in Postgres...")
                subprocess.run(["python", "scripts/embed_chunks.py"], check=True)

                print("→ Rebuilding BM25 index...")
                subprocess.run(["python", "scripts/build_bm25_index.py"], check=True)

                print(f"Update complete for {path}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error during update: {e}")

def start_watcher():
    observer = Observer()
    handler = ChangeHandler()
    for d in WATCH_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
        observer.schedule(handler, d, recursive=True)
        print(f"Watching {d}")
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    print("\n=== EMBEDDING WATCHER STARTED ===\n")
    start_watcher()
