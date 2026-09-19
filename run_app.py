"""
Main runner script for RAG Error Decomposition & Failure Analysis.
Usage:
    python run_app.py           # Launches the Streamlit Dashboard
    python run_app.py --api     # Launches the FastAPI Backend API
    python run_app.py --test    # Runs automated end-to-end unit tests
"""

import sys
import argparse
import subprocess


def run_dashboard():
    """Runs the Streamlit web dashboard."""
    print("[RUN] Launching Streamlit Research Dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"])


def run_api():
    """Runs the FastAPI server with Uvicorn."""
    print("[API] Launching FastAPI Backend on http://127.0.0.1:8000 ...")
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"])


def run_tests():
    """Runs the automated test suite."""
    print("[TEST] Running Unit & Integration Tests...")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Error Decomposition & Failure Analysis Runner")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI backend")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    args = parser.parse_args()

    if args.api:
        run_api()
    elif args.test:
        run_tests()
    else:
        run_dashboard()
