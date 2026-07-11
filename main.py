import os
import time
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types
from google.genai.errors import APIError

app = FastAPI()

# Initialize the Gemini Client
client = genai.Client()

# --- Request/Response Schemas ---

class ProblemRequest(BaseModel):
    problem_id: str
    problem: str

class SolverResponse(BaseModel):
    reasoning: str = Field(..., description="Step-by-step mathematical reasoning.")
    answer: int = Field(..., description="The final single integer answer.")

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning_length(cls, v: str) -> str:
        if len(v) < 80:
            raise ValueError("Reasoning must be at least 80 characters long.")
        return v

    @field_validator("answer")
    @classmethod
    def validate_integer(cls, v: any) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("Answer must be a strict JSON integer.")
        return v

# --- Helper function to handle 429 Rate Limits gracefully ---

def generate_content_with_retry(contents, config, max_retries: int = 4, initial_delay: float = 1.0):
    """
    Wraps the content generation call with tighter exponential backoff 
    to prevent exceeding the grader's 25-second timeout window.
    """
    delay = initial_delay
    for i in range(max_retries):
        try:
            resp = client.models.generate_content(
                model='gemini-1.5-flash-8b',  # High-throughput model tier with larger quotas
                contents=contents,
                config=config,
            )
            return resp
        except APIError as e:
            if getattr(e, 'code', None) == 429 or "429" in str(e):
                if i == max_retries - 1:
                    raise e
                time.sleep(delay)
                delay *= 1.5  # Gentler pacing multiplication to keep total time low
            else:
                raise e
    raise HTTPException(status_code=500, detail="Failed after maximum retries due to rate limits.")

# --- Routes ---

@app.get("/")
async def root():
    return {"status": "healthy", "message": "Word solver engine with rate-limit protection is awake!"}

@app.post("/solve", response_model=SolverResponse)
async def solve_word_problem(request: ProblemRequest):
    try:
        system_instruction = (
            "You are a precise mathematical solver. Your job is to solve the given word problem.\n"
            "Rules:\n"
            "1. Identify and completely ignore irrelevant numbers or background distractor data.\n"
            "2. Break down the calculation step-by-step.\n"
            "3. Ensure the written reasoning explanation is thorough, explicit, and exceeds 80 characters.\n"
            "4. Compute the final answer as a single, strict integer."
        )

        # Generate structured content using the retry wrapper
        response = generate_content_with_retry(
            contents=f"Problem: {request.problem}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=SolverResponse,
            )
        )

        validated_response = response.parsed

        if not validated_response:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, 
                detail="The model failed to return a conforming response."
            )

        return validated_response

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation failed: {str(val_err)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )