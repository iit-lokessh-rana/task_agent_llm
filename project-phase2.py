from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel
import os
import logging
from fastapi.security.api_key import APIKeyHeader
import json
import queue
import threading
import ssl
import subprocess
import shutil
import sqlite3
import requests
import markdown
import csv
from datetime import datetime
from final_util import chat_completion, generate_embeddings

# Ensure SSL is available
try:
    ssl.create_default_context()
except ImportError:
    raise RuntimeError("SSL module is missing. Ensure that Python is compiled with SSL support.")

app = FastAPI()

# Root Route
@app.get("/")
def read_root():
    return {"message": "Welcome to your FastAPI Automation Agent!"}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Key Authentication
API_KEY = os.getenv("API_KEY", "your_secure_api_key")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

class TaskRequest(BaseModel):
    task: str

# Security Constraints
DATA_DIR = "/data"

def enforce_security_constraints(filepath):
    if not filepath.startswith(DATA_DIR):
        raise HTTPException(status_code=403, detail="Access outside /data is not allowed.")
    if os.path.exists(filepath) and os.path.isdir(filepath):
        raise HTTPException(status_code=403, detail="Access to directories is restricted.")

def execute_task(parsed_task: dict):
    """Executes the parsed task while enforcing security constraints."""
    try:
        action = parsed_task.get("action")
        
        if action == "fetch_api_data":
            fetch_api_data(parsed_task.get("url"), parsed_task.get("output"))
        elif action == "clone_git_repo":
            clone_git_repo(parsed_task.get("repo_url"), parsed_task.get("commit_message"))
        elif action == "run_sql_query":
            run_sql_query(parsed_task.get("db_path"), parsed_task.get("query"), parsed_task.get("output"))
        elif action == "scrape_website":
            scrape_website(parsed_task.get("url"), parsed_task.get("output"))
        elif action == "compress_image":
            compress_image(parsed_task.get("input"), parsed_task.get("output"))
        elif action == "transcribe_audio":
            transcribe_audio(parsed_task.get("input"), parsed_task.get("output"))
        elif action == "convert_md_to_html":
            convert_md_to_html(parsed_task.get("input"), parsed_task.get("output"))
        elif action == "filter_csv":
            filter_csv(parsed_task.get("input"), parsed_task.get("output"), parsed_task.get("criteria"))
    except Exception as e:
        logger.error(f"Task execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Task execution error: {str(e)}")

def fetch_api_data(url, output_file):
    enforce_security_constraints(output_file)
    response = requests.get(url)
    with open(output_file, "w") as f:
        f.write(response.text)

def clone_git_repo(repo_url, commit_message):
    subprocess.run(["git", "clone", repo_url, f"{DATA_DIR}/repo"], check=True)
    with open(f"{DATA_DIR}/repo/commit.txt", "w") as f:
        f.write(commit_message)

def run_sql_query(db_path, query, output_file):
    enforce_security_constraints(db_path)
    enforce_security_constraints(output_file)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

def scrape_website(url, output_file):
    enforce_security_constraints(output_file)
    response = requests.get(url)
    with open(output_file, "w") as f:
        f.write(response.text)

def compress_image(input_file, output_file):
    enforce_security_constraints(input_file)
    enforce_security_constraints(output_file)
    subprocess.run(["convert", input_file, "-resize", "50%", output_file], check=True)

def transcribe_audio(input_file, output_file):
    enforce_security_constraints(input_file)
    enforce_security_constraints(output_file)
    subprocess.run(["whisper", input_file, "--output", output_file], check=True)

def convert_md_to_html(input_file, output_file):
    enforce_security_constraints(input_file)
    enforce_security_constraints(output_file)
    with open(input_file, "r") as f:
        md_content = f.read()
    html_content = markdown.markdown(md_content)
    with open(output_file, "w") as f:
        f.write(html_content)

def filter_csv(input_file, output_file, criteria):
    enforce_security_constraints(input_file)
    enforce_security_constraints(output_file)
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        filtered_rows = [row for row in reader if eval(criteria)]
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=filtered_rows[0].keys())
        writer.writeheader()
        writer.writerows(filtered_rows)

@app.post("/run", dependencies=[Depends(verify_api_key)])
def run_task(task: TaskRequest, background_tasks: BackgroundTasks):
    """Parses and executes a task with strict security enforcement."""
    try:
        if not task.task:
            raise HTTPException(status_code=400, detail="Task description cannot be empty.")
        structured_task = chat_completion(task.task)
        execute_task(structured_task)
        return {"status": "completed", "message": "Task executed successfully."}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
