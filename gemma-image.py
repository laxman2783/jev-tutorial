import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel(model_name='gemma3')
response = model.generate_content('Teach me about how an LLM works')

print(response.text)
