import os
from groq import Groq
from dotenv import load_dotenv
from app.retriever import retrieve
from app.tools import get_air_quality

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# -------- Improved City Extraction --------
def extract_city(query):
    words = query.lower().split()

    if "in" in words:
        idx = words.index("in")
        if idx + 1 < len(words):
            return words[idx + 1].capitalize()

    return words[-1].capitalize()


def run_agent(messages):
    user_query = messages[-1]["content"]
    query_lower = user_query.lower()

    # -------- Tool decision --------
    use_tool = any(word in query_lower for word in [
        "air quality", "pollution", "aqi"
    ])

    tool_result = ""

    if use_tool:
        city = extract_city(user_query)
        tool_result = get_air_quality(city)

    # -------- RAG --------
    retrieved_data = retrieve(user_query)

    context = "\n\n".join([d["text"] for d in retrieved_data])
    sources = list(set([d["source"] for d in retrieved_data]))

    # -------- Prompt --------
    system_prompt = f"""
    You are an intelligent environmental assistant.

    Use:
    - Context for explanations (ONLY from provided text)
    - Tool data for real-time info

    Context:
    {context}

    Tool Data:
    {tool_result}

    Rules:
    - Use ONLY the provided context for facts
    - DO NOT add external sources like CDC, WHO unless explicitly in context
    - DO NOT create fake sections like "Real-time tool data"
    - If tool data exists, use it clearly
    - If information is missing, say "not available in provided data"
    - Keep answer clean and structured
    """


    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt}] + messages,
    )

    answer = response.choices[0].message.content

    # -------- Append sources --------
    if sources:
        answer += "\n\n📚 Sources:\n"
        for s in sources:
            answer += f"- {s}\n"

    return answer
