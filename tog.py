from together import Together
from dotenv import load_dotenv
import os
load_dotenv()
client = Together(api_key=os.environ.get("TOGETHER_API_KEY")) # auth defaults to os.environ.get("TOGETHER_API_KEY")

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
      {
        "role": "user",
        "content": "What are some fun things to do in New York?"
      }
    ]
)
print(response.choices[0].message.content)