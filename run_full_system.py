import subprocess
import time
import os
import sys
import signal
import socket
import requests

def start_process(command, cwd=None):
    # Start process in a new session for easier group termination
    return subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
        cwd=cwd,
        text=True  # capture output as strings
    )

def wait_for_tcp(host, port, timeout=60, name="service"):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"{name} is reachable on {host}:{port}")
                return True
        except OSError:
            time.sleep(1)
    print(f"Timeout waiting for {name} on {host}:{port}")
    return False

def wait_for_http(url, timeout=60, name="service"):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                print(f"{name} HTTP ready at {url}")
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    print(f"Timeout waiting for {name} HTTP at {url}")
    return False

def build_llama_cpp():
    print("Building llama.cpp with CMake...")
    repo_path = "./llama.cpp"
    try:
        subprocess.check_call("mkdir -p build", shell=True, cwd=repo_path)
        subprocess.check_call("cmake ..", shell=True, cwd=os.path.join(repo_path, "build"))
        subprocess.check_call("cmake --build .", shell=True, cwd=os.path.join(repo_path, "build"))
        print("llama.cpp built successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error building llama.cpp: {e}")
        sys.exit(1)

def tail_proc(prefix, proc, max_lines=20):
    # Print last lines of stdout/stderr if a process fails to become healthy
    try:
        out, err = proc.communicate(timeout=0.2)
    except Exception:
        out, err = "", ""
    out_lines = [l for l in out.splitlines() if l.strip()][-max_lines:]
    err_lines = [l for l in err.splitlines() if l.strip()][-max_lines:]
    if out_lines:
        print(f"\n[{prefix}] stdout (tail):")
        for l in out_lines:
            print(l)
    if err_lines:
        print(f"\n[{prefix}] stderr (tail):")
        for l in err_lines:
            print(l)

def main():
    # Build llama.cpp
    build_llama_cpp()

    # Commands and ports
    llama_server_path = "./llama.cpp/bin/llama-server"
    model_path = "/media/rishikesh/Rishi/RAGBOT/rag-banking/models/Qwen2.5-7B-Instruct-Q6_K.gguf"

    llama_cmd = f'{llama_server_path} -m "{model_path}" -ngl 5 --port 8080'
    backend_cmd = "uvicorn api.server:app --host 0.0.0.0 --port 8000"
    embedding_watcher_cmd = "python scripts/embedding_watcher.py"
    auto_ingest_cmd = "python scripts/auto_ingest.py"
    # Streamlit must bind to all interfaces and fixed port
    streamlit_cmd = "streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"

    # Start llama-server
    print("Starting llama-server...")
    llama_proc = start_process(llama_cmd)
    if not wait_for_tcp("127.0.0.1", 8080, timeout=90, name="llama-server"):
        tail_proc("llama-server", llama_proc)
        print("Exiting due to llama-server not ready.")
        os.killpg(os.getpgid(llama_proc.pid), signal.SIGTERM)
        sys.exit(1)

    # Start FastAPI backend (on port 8000)
    print("Starting FastAPI backend...")
    backend_proc = start_process(backend_cmd)
    if not wait_for_tcp("127.0.0.1", 8000, timeout=60, name="backend"):
        tail_proc("backend", backend_proc)
        print("Exiting due to backend not ready.")
        os.killpg(os.getpgid(backend_proc.pid), signal.SIGTERM)
        os.killpg(os.getpgid(llama_proc.pid), signal.SIGTERM)
        sys.exit(1)

    # Start embedding watcher
    print("Starting embedding watcher...")
    embed_proc = start_process(embedding_watcher_cmd)
    time.sleep(2)

    # Start auto-ingestion pipeline
    print("Starting auto-ingestion pipeline...")
    ingest_proc = start_process(auto_ingest_cmd)
    time.sleep(2)

    # Start Streamlit UI
    print("Starting Streamlit UI...")
    streamlit_proc = start_process(streamlit_cmd)
    # Wait for Streamlit to open HTTP (8501). It serves a root page.
    if not wait_for_http("http://127.0.0.1:8501/", timeout=60, name="Streamlit"):
        tail_proc("streamlit", streamlit_proc)
        print("Continuing even if Streamlit not confirmed; check logs.")

    print("All services started.")
    print("Endpoints:")
    print(" - Llama.cpp UI:   http://127.0.0.1:8080")
    print(" - FastAPI /ask:   http://127.0.0.1:8000/ask")
    print(" - Streamlit UI:   http://127.0.0.1:8501")
    print("Press Ctrl+C to stop everything.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating all subprocesses...")
        for proc in [streamlit_proc, ingest_proc, embed_proc, backend_proc, llama_proc]:
            if proc and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        sys.exit(0)

if __name__ == "__main__":
    main()
