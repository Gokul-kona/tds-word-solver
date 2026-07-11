import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from google import genai

load_dotenv()

app = FastAPI()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


class Problem(BaseModel):
    problem_id: str
    problem: str


PROMPT = """
You are an expert arithmetic word-problem solver.

Solve the user's problem carefully.

Return ONLY valid JSON with EXACTLY these two keys:

{
  "reasoning": "A concise explanation of the calculation in at least 80 characters. Do not reveal hidden reasoning or internal deliberation. Simply explain the arithmetic performed.",
  "answer": 123
}

Rules:
- answer must be a JSON integer.
- Do NOT return a string or float.
- Ignore irrelevant numbers.
- No markdown.
- No extra keys.
"""


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/solve")
def solve(req: Problem):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"{PROMPT}\n\nProblem:\n{req.problem}",
    )

    text = response.text.strip()

    # Remove markdown if Gemini adds it
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    data = json.loads(text)

    # Validate output
    if set(data.keys()) != {"reasoning", "answer"}:
        raise ValueError("Model returned incorrect keys")

    if not isinstance(data["reasoning"], str):
        raise ValueError("Reasoning must be a string")

    if len(data["reasoning"]) < 80:
        raise ValueError("Reasoning is too short")

    if not isinstance(data["answer"], int):
        raise ValueError("Answer must be an integer")

    return data