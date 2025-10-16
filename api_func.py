import os
from dotenv import load_dotenv
import openai

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# OpenAI
def stream_openai_response(prompt: str):
    try:
        stream = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield {"data": delta.content}

        yield {"event": "end", "data": ""}
    except Exception as e:
        yield {"event": "error", "data": str(e)}


def get_openai_response(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        stream=False,
    )

    return response.choices[0].message.content.strip()
