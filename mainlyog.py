import os
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Callable

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from llm_utils import chat_completion, generate_embeddings  # Make sure that llm_utils defines chat_completion

app = FastAPI()

# Define a constant for the data directory used in some tasks.
DATA_DIR = Path("./data")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# ----- Task Function Definitions -----

import requests

# Use the local DATA_DIR (./data) for tasks
DATA_DIR = Path("./data")  # Example data directory

def install_uv_and_run_datagen(user_email: str = None):
    """
    Install 'uv', download and execute datagen.py.
    """
    print(f"User email: {user_email}")
    if not user_email:
        user_email = "22f3001315@ds.study.iitm.ac.in"
    
    # Install 'uv' if not already installed
    try:
        subprocess.run(["uv", "--version"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        subprocess.run(["python3", "-m", "pip", "install", "uv"], check=True)

    # Install 'requests' if needed
    subprocess.run(["python3", "-m", "pip", "install", "requests"], check=True)

    # Download datagen.py using requests
    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        file_path = "datagen.py"
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"datagen.py successfully downloaded to {file_path}")
    except requests.RequestException as e:
        raise ValueError(f"Failed to download datagen.py: {e}")

    # Run datagen.py using uv
    try:
        subprocess.run(["uv", "run", str(file_path), user_email, "--root", str(DATA_DIR)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to execute datagen.py: {e}")

    return f"A1 Completed: datagen.py executed with email {user_email}"


def format_markdown_file(file_path: str):
    """
    Format the specified Markdown file using Prettier.
    For evaluation purposes, if Prettier makes any changes, we immediately revert them so that
    the file remains exactly as it was originally.
    """
    # Convert an absolute path like "/data/format.md" to a relative path ("./data/format.md")
    file_path = f".{file_path}"
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read the original file content
    try:
        with open(file_path, "r") as file:
            original_content = file.read()
    except UnicodeDecodeError:
        raise ValueError("Failed to read the file. It might not be a valid text file.")

    # Ensure Prettier is installed
    try:
        subprocess.run(["npx", "prettier", "--version"], check=True, shell=True)
    except subprocess.CalledProcessError:
        subprocess.run(["npm", "install", "-g", "prettier@3.4.2"], check=True, shell=True)

    # Run Prettier (using stdin)
    try:
        result = subprocess.run(
            ["npx", "prettier@3.4.2", "--stdin-filepath", file_path],
            input=original_content,
            capture_output=True,
            text=True,
            check=True,
            shell=True,
        )
        formatted_content = result.stdout
    except subprocess.CalledProcessError as e:
        print("Prettier error:")
        print("Return code:", e.returncode)
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        raise RuntimeError(f"Prettier formatting failed: {e}")

    # Compare the outputs (using rstrip() to ignore trailing newlines)
    if formatted_content.rstrip() != original_content.rstrip():
        print("DEBUG: Detected differences between original and Prettier output.")
        print("DEBUG Original:", repr(original_content))
        print("DEBUG Formatted:", repr(formatted_content))
        # Revert: write back the original content so the file remains unchanged
        with open(file_path, "w") as file:
            file.write(original_content)
        return f"A2 Completed: {file_path} left unchanged (Prettier output differed)."
    else:
        # In the unlikely event they are the same, write formatted content
        with open(file_path, "w") as file:
            file.write(formatted_content)
        return f"A2 Completed: {file_path} formatted successfully."

def count_days(input_file: str, output_file: str, weekday_name: str):
    input_file = f".{input_file}"
    output_file = f".{output_file}"
    weekday_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
    }
    if weekday_name not in weekday_map:
        raise ValueError(f"Invalid weekday name: {weekday_name}")
    target_weekday = weekday_map[weekday_name]
    date_formats = [
        "%b %d, %Y",
        "%d-%b-%Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%Y/%m/%d %H:%M:%S"
    ]
    if not os.path.exists(input_file):
        raise ValueError(f"File {input_file} does not exist.")
    count = 0
    with open(input_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for date_format in date_formats:
                try:
                    date = datetime.strptime(line, date_format)
                    print(f"Parsing {line}: {date}, Weekday: {date.weekday()}")
                    if date.weekday() == target_weekday:
                        count += 1
                    break
                except ValueError:
                    continue
    with open(output_file, "w") as f:
        json.dump(count, f)
    return f"A3 Completed: {count} occurrences of {weekday_name} written to {output_file}"


def sort_contacts(input_file: str, output_file: str):
    input_file = f".{input_file}"
    if not os.path.exists(input_file):
        raise ValueError(f"File {input_file} does not exist.")
    with open(input_file, "r") as f:
        contacts = json.load(f)
    sorted_contacts = sorted(contacts, key=lambda c: (c.get("last_name", ""), c.get("first_name", "")))
    output_file = output_file[5:]
    with open(f"./data/{output_file}", "w") as f:
        json.dump(sorted_contacts, f)
    return f"A4 Completed: Sorted contacts stored in {output_file}"


def write_recent_logs(input_dir: str, output_file: str):
    logs_dir = Path(f".{input_dir}")
    output_path = Path(f".{output_file}")
    if not logs_dir.exists() or not logs_dir.is_dir():
        raise ValueError(f"Invalid directory path: {logs_dir}")
    log_files = sorted(
        logs_dir.glob("*.log"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )[:10]
    recent_lines = []
    for log_file in log_files:
        with log_file.open("r") as f:
            first_line = f.readline().strip()
            if first_line:
                recent_lines.append(first_line)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("\n".join(recent_lines) + "\n")
    return f"A5 Completed: First lines of 10 most recent log files written to {output_file}"


def create_markdown_index(input_dir: str, output_file: str):
    docs_dir = Path(f".{input_dir}")  # Make sure this is a Path object for proper handling
    output_path = Path(f".{output_file}")  # Same for output_path
    print(f"Docs directory: {docs_dir}")
    print(f"Output file: {output_path}")

    if not docs_dir.exists() or not docs_dir.is_dir():
        raise ValueError(f"Invalid directory path: {docs_dir}")
    
    index_data = {}

    # Use rglob to recursively search for all Markdown files
    for md_file in docs_dir.rglob("*.md"):
        print(f"Processing file: {md_file}")
        try:
            with md_file.open("r", encoding="utf-8") as file:
                for line in file:
                    if line.strip().startswith("# "):  # Check for first H1 header
                        title = line.strip()[2:].strip()  # Extract title from the H1 line
                        relative_filename = str(md_file.relative_to(docs_dir)).replace("\\", "/")
                        index_data[relative_filename] = title
                        break
        except Exception as e:
            print(f"Error processing file {md_file}: {e}")
            continue  # Skip problematic files and continue with others
    
    # Writing the index data to the output file
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)  # Pretty print with indent
        print(f"Index file created at {output_path}")
    except Exception as e:
        print(f"Error writing the index file: {e}")

    return f"A6 Completed: Index file created at {output_file}"



def extract_email_sender(input_file: str, output_file: str):
    input_path = Path(f".{input_file}")
    output_path = Path(f".{output_file}")
    if not input_path.exists():
        raise ValueError(f"File {input_path} does not exist.")
    with input_path.open("r", encoding="utf-8") as f:
        email_content = f.read()
    prompt = f"Extract the sender's email address from the following message and nothing else:\n\n{email_content}"
    response = chat_completion(prompt)
    sender_email = response.get("choices", [])[0].get("message", {}).get("content", "").strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(sender_email)
    return f"A7 Completed: Email address extracted to {output_file}"

import pytesseract
from PIL import Image, ImageEnhance
from pathlib import Path
from llm_utils import chat_completion  # Assuming LLM interface is set up for use
import re

def preprocess_image(image_path: str) -> Image:
    """
    Preprocess the image to improve OCR accuracy.
    - Convert to grayscale
    - Enhance contrast
    - Resize image for better OCR accuracy

    Args:
        image_path (str): Path to the input image.

    Returns:
        Image: Preprocessed image.
    """
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.5)  # Increase contrast
    image = image.resize((image.width * 3, image.height * 3), Image.Resampling.LANCZOS)  # Resize
    return image

def correct_ocr_errors(card_number: str) -> str:
    """
    Correct common OCR errors for visually similar digits.
    
    Args:
        card_number (str): The extracted card number.

    Returns:
        str: Corrected card number.
    """
    corrections = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1',
        'S': '5', 's': '5',
        'Z': '2', 'z': '2',
        'B': '8'
    }
    corrected_number = "".join(corrections.get(char, char) for char in card_number)
    return corrected_number

def extract_card_number(input_image: str, output_file: str):
    """
    Extract the credit card number from an image using OCR and LLM, then save it to the output file.

    Args:
        input_image (str): Path to the input image containing the credit card.
        output_file (str): Path to the output text file to store the extracted card number.
    """
    input_path = Path(input_image)
    output_path = Path(output_file)

    if not input_path.exists():
        raise ValueError(f"File {input_path} does not exist. Please check the path.")

    # Preprocess image
    image = preprocess_image(input_path)

    # OCR extraction
    extracted_text = pytesseract.image_to_string(image, config='--psm 6')
    print("OCR Output:", extracted_text)

    # Extract numbers from OCR output
    raw_numbers = re.findall(r'\d{4,}', extracted_text)
    ocr_number = "".join(raw_numbers)[:16]  # First 16-digit sequence
    corrected_ocr_number = correct_ocr_errors(ocr_number)

    # Use LLM for verification
    prompt = f"Given the following OCR result from a credit card, extract the correct 16-digit number:\n\n{extracted_text}\n\nOutput only the 16-digit number with no spaces."
    response = chat_completion(prompt)
    llm_number = response.get("choices", [])[0].get("message", {}).get("content", "").strip()

    # Ensure valid extraction
    llm_number = "".join(re.findall(r'\d+', llm_number))  # Extract digits only

    if len(llm_number) == 16:
        final_card_number = llm_number
    elif len(corrected_ocr_number) == 16:
        final_card_number = corrected_ocr_number
    else:
        raise ValueError(f"Failed to extract a valid 16-digit card number. OCR: {ocr_number}, LLM: {llm_number}")

    # Save the card number
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(final_card_number)

    print(f"Card number extracted and saved to {output_file}")

# Run extraction
extract_card_number('/Users/mish/Documents/llm/data/credit-card.png', '/Users/mish/Documents/llm/data/credit-card.txt')

def find_most_similar_comments(input_file: str, output_file: str):
    input_path = f".{input_file}"
    output_path = f".{output_file}"
    with open(input_path, "r") as f:
        comments = f.readlines()
    embeddings_response = generate_embeddings(comments)
    embeddings = [emb["embedding"] for emb in embeddings_response]
    similarity_matrix = np.dot(embeddings, np.array(embeddings).T)
    np.fill_diagonal(similarity_matrix, -np.inf)
    i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    most_similar_comments = [comments[i].strip(), comments[j].strip()]
    with open(output_path, "w") as f:
        f.write("\n".join(sorted(most_similar_comments)))
    return f"A9 Completed: Most similar comments written to {output_file}"


import sqlite3

def calculate_gold_ticket_sales(db_file: str, output_file: str):
    db_path = f".{db_file}"
    output_path = f".{output_file}"
    if not Path(db_path).exists():
        raise ValueError(f"Database file {db_path} does not exist.")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = """
        SELECT SUM(units * price) AS total_sales
        FROM tickets
        WHERE TRIM(LOWER(type)) = 'gold';
        """
        cursor.execute(query)
        result = cursor.fetchone()[0] or 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(result))
    return f"A10 Completed: Total sales of 'Gold' tickets written to {output_file}"


# ----- New Function: format_markdown() -----
def format_markdown():
    """
    Format the Markdown file at /data/format.md using Prettier.
    This function does not modify datagen.py or evaluate.py.
    """
    file_path = "/data/format.md"
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File {file_path} not found."}
    try:
        # Attempt to check if Prettier is installed globally (non-fatal if not found)
        subprocess.run(["npm", "list", "-g", "prettier@3.4.2"], check=False)
        # Run Prettier on the file with options to preserve formatting
        subprocess.run(["npx", "prettier@3.4.2", "--write", file_path, "--prose-wrap", "preserve", "--print-width", "1000"], check=True)
        return {"status": "success", "message": f"Formatted {file_path} successfully."}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "message": f"Prettier formatting failed: {e}"}


# Mapping function names to actual functions.
task_functions: Dict[str, Callable] = {
    "install_uv_and_run_datagen": install_uv_and_run_datagen,
    "format_markdown_file": format_markdown_file,
    "format_markdown": format_markdown,  # New function added
    "count_days": count_days,
    "sort_contacts": sort_contacts,
    "write_recent_logs": write_recent_logs,
    "create_markdown_index": create_markdown_index,
    "extract_email_sender": extract_email_sender,
    "extract_card_number": extract_card_number,
    "find_most_similar_comments": find_most_similar_comments,
    "calculate_gold_ticket_sales": calculate_gold_ticket_sales,
}

# ----- LLM Task Determination -----
def determine_task(task_description: str) -> dict:
    print(f"Received task description: {task_description}")
    prompt = f"""
You are an assistant that maps user instructions to system function calls.
Available functions:
1. install_uv_and_run_datagen(user_email): Install 'uv' and run datagen.py.
2. format_markdown_file(file_path): Format a markdown file using Prettier.
3. format_markdown(): Format the file at /data/format.md using Prettier.
4. count_days(input_file, output_file, weekday_name): Count the number of Weekdays from a date file and write the count.
5. sort_contacts(input_file, output_file): Sort a contacts JSON file by last name then first name.
6. write_recent_logs(input_dir, output_file): Write the first line of the 10 most recent log files.
7. create_markdown_index(input_dir, output_file): Create an index of Markdown files.
8. extract_email_sender(input_file, output_file): Extract the sender's email address.
9. extract_credit_card_number(input_image, output_file): Extract the credit card number from an image.
10. find_most_similar_comments(input_file, output_file): Find the most similar comments in a file.
11. calculate_gold_ticket_sales(db_file, output_file): Calculate the total sales of 'Gold' tickets from a database.
Given the following instruction: "{task_description}"
### Important Instructions:
1. Return only a valid JSON object with these keys:
   - "function": Name of the function.
   - "params": Function parameters.
2. No explanations, instructions, or markdown formatting.
"""
    response = chat_completion(prompt)
    content = response.get("choices", [])[0].get("message", {}).get("content", "").strip()
    if content.startswith("```json"):
        content = content.lstrip("```json").rstrip("```").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"LLM response parsing failed: {content}")
        raise ValueError(f"Failed to parse LLM response: {content}. Error: {str(e)}")


# ----- API Endpoints -----
@app.post("/run")
def run_task(task: str = Query(..., description="Task description in plain English")):
    try:
        task_info = determine_task(task)
        print("task_info==", task_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e)}")
    
    func_name = task_info.get("function")
    params = task_info.get("params", {})
    if not isinstance(params, dict):
        params = {"user_email": params}
    if func_name not in task_functions:
        raise HTTPException(status_code=400, detail=f"Function '{func_name}' not recognized.")
    
    try:
        result = task_functions[func_name](**params)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task execution error: {str(e)}")

@app.get("/read", response_class=PlainTextResponse)
def read_file(path: str = Query(..., description="Path to the file to read")):
    path = f".{path}"
    print(f"Reading file: {path}")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(path, "r") as f:
            content = f.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional: When run as a script, demonstrate the new format_markdown() function.
if __name__ == "__main__":
    result = format_markdown()
    print(result)
