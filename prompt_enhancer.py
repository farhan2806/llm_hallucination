import google.generativeai as genai
import streamlit as st

# Configure Gemini API
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

SYSTEM_INSTRUCTION = """
You are a world-class expert in Prompt Engineering for Large Language Models (LLMs).
Your task is to take a raw, often vague user input and rewrite it into a highly effective, structured, and optimized prompt.

Follow these principles:
1. Clarity & Specificity: Remove ambiguity. Define what needs to be done clearly.
2. Structure: Use sections like [Role], [Task], [Context], [Constraints], and [Output Format] if the complexity warrants it.
3. Persona: Assign a relevant expert persona (e.g., "Act as a senior software engineer...").
4. Output Definition: Clearly state how the result should look (e.g., "markdown table", "JSON", "bullet points").
5. Conciseness: Do not add unnecessary fluff, but ensure all necessary context is present.

IMPORTANT: Return ONLY the enhanced prompt text. Do not add conversational filler like "Here is your enhanced prompt:".
"""

model = genai.GenerativeModel("gemini-flash-latest")

def enhance_prompt_with_gemini(raw_prompt: str):

    response = model.generate_content(
        f"{SYSTEM_INSTRUCTION}\n\nUser Input:\n{raw_prompt}",
        generation_config={"temperature": 0.7},
    )

    if not response.text:
        raise Exception("No response generated from Gemini.")

    return response.text.strip()
