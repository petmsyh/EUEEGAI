from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import pickle
from sentence_transformers import SentenceTransformer
from vector_search import local_similarity_search
import google.generativeai as genai
import re
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_URL = os.getenv("GEMNIE_API_KEY")

# --- Gemini API Setup ---
genai.configure(api_key=API_URL)
model_gemini = genai.GenerativeModel("gemini-2.0-flash")

# --- Load model and points ---
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
with open("./asset_/Geography_history_questions_points.pkl", "rb") as f:
    points = pickle.load(f)
print(f"Loaded model and {len(points)} points.")

def build_gemini_prompt(results):
    context = ""
    for i, r in enumerate(results, 1):
        options = r['options'] if isinstance(r['options'], list) else []
        while len(options) < 4:
            options.append("[Missing Option]")
        source = r.get('source', '[Source not available]')
        context += (
            f"Question {i}: {r['text']}\n"
            f"A) {options[0]}\n"
            f"B) {options[1]}\n"
            f"C) {options[2]}\n"
            f"D) {options[3]}\n"
            f"Source: {source}\n\n"
        )
    prompt = f"""You are a smart educational assistant. Given the following context of multiple-choice questions, do the following for each question:

- **Never skip any question**. For every question in the context, always return a result.
- If the question text, options, correct answer, or explanation is missing or incomplete, deduce and fill it in based on the available information.
- For each question, return:
    - The full, improved question text
    - Make it to return I don't have any questions related to this query if no relevant questions matching to the query are found.
    - A list of clearly labeled multiple-choice options (A, B, C, D)
    - The correct answer clearly stated (with the corresponding letter)
    - A clear explanation of at least 3 lines justifying the correct answer
    - The source (as provided in the context)

Question 1: 
[Full question text]

A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

Correct Answer: [Letter]
Explanation: [At least 3 lines]

Source: [Source]

[Repeat for all questions. Do not skip any question.]

Questions:
{context}
"""
    return prompt

def extract_limit_from_query(query, default=10, max_limit=None):
    match = re.search(r"\b(\d+)\b", query)
    if match:
        limit = int(match.group(1))
        if max_limit:
            limit = min(limit, max_limit)
        return limit
    for word, num in {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20
    }.items():
        if re.search(rf"\b{word}\b", query, re.IGNORECASE):
            limit = num
            if max_limit:
                limit = min(limit, max_limit)
            return limit
    return default

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/answer")
async def get_answer(request: QueryRequest):
    query = request.query
    max_limit = len(points)
    limit = extract_limit_from_query(query, default=10, max_limit=max_limit)

    # Subject filtering
    subjects = ["history", "geography", "maths", "mathematics", "civics", "biology", "chemistry", "physics", "english", "aptitude"]
    subject_in_query = next((subj for subj in subjects if subj.lower() in query.lower()), None)

    results = local_similarity_search(query, model, points, limit=limit)
    filtered_results = [
        r for r in results
        if subject_in_query and subject_in_query in r.get('source', '').lower()
    ] if subject_in_query else results

    if not filtered_results:
        return {"answer": "I don't have any questions related to this query."}
    else:
        prompt = build_gemini_prompt(filtered_results)
        try:
            response = model_gemini.generate_content(prompt)
            return {"answer": response.text.strip()}
        except Exception as e:
            return {"answer": f"Error generating response: {e}"}

@app.get("/")
def read_root():
    return {"message": "EUEE AI backend is running!"}
