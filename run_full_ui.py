import subprocess
import time
import os
import signal
import sys

def start_process(cmd, cwd=None):
    # Start the process in its own process group for easier termination
    return subprocess.Popen(cmd, shell=True, cwd=cwd, preexec_fn=os.setsid)

def main():
    print("Starting FastAPI backend...")
    backend_cmd = "uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload"
    backend_proc = start_process(backend_cmd)
    time.sleep(5)  # wait a bit for backend to be ready

    # Before starting Streamlit, check if port 8501 is free, or change port accordingly
    print("Starting Streamlit UI...")
    streamlit_cmd = "streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"

    streamlit_proc = start_process(streamlit_cmd)

    print("Both backend and UI running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping backend and UI...")
        for proc in [streamlit_proc, backend_proc]:
            if proc and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        sys.exit(0)

if __name__ == "__main__":
    main()
