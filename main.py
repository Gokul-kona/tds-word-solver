import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from google import genai
from google.genai import types

app = FastAPI()

# Initialize Gemini Client (automatically reads GEMINI_API_KEY environment variable)
client = genai.Client()

# --- Request/Response Schemas (Strict Grader Specifications) ---

class ProblemRequest(BaseModel):
    problem_id: str
    problem: str

class SolverResponse(BaseModel):
    reasoning: str = Field(..., description="Step-by-step mathematical reasoning.")
    answer: int = Field(..., description="The final single integer answer.")

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning_length(cls, v: str) -> str:
        # Grader Rule: reasoning must be >= 80 characters
        if len(v) < 80:
            raise ValueError("Reasoning must be at least 80 characters long.")
        return v

    @field_validator("answer")
    @classmethod
    def validate_integer(cls, v: any) -> int:
        # Grader Rule: Strict JSON integer (no floats, no booleans, no strings)
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError("Answer must be a strict JSON integer.")
        return v


# --- API Endpoint ---

@app.post("/solve", response_model=SolverResponse)
async def solve_word_problem(request: ProblemRequest):
    try:
        # Instruction ensuring distractors are ignored and character limits met
        system_instruction = (
            "You are a precise mathematical solver. Your job is to solve the given word problem.\n"
            "Rules:\n"
            "1. Identify and completely ignore irrelevant numbers or background distractor data.\n"
            "2. Break down the calculation step-by-step.\n"
            "3. Ensure the written reasoning explanation is thorough, explicit, and exceeds 80 characters.\n"
            "4. Compute the final answer as a single, strict integer."
        )

        # Generate structured content using Gemini
        response = client.models.generate_content(
            model='gemini-3.5-flash', # Blazing fast, ideal for programmatic workflows
            contents=f"Problem: {request.problem}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0, # Forces deterministic calculation behavior
                response_mime_type="application/json",
                response_schema=SolverResponse, # Direct native Pydantic enforcement
            ),
        )

        # The SDK automatically handles parsing into the schema when response_schema is used
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