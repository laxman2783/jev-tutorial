from ollama import chat

# Read the cover letter text
with open("cover-letter.txt", "r", encoding="utf-8") as f:
    content = f.read()

SYSTEM_PROMPT =f'you are experience in language translator. translate the text {content} into spanish language.Keep the language polite and professional. Do not provide any extra information.Do not provide any choices'
# Ask Gemma 3 to translate the text to Spanish
response = chat(
    model="gemma3:1b",
    messages=[
        {
            "role": "user",
            "content": SYSTEM_PROMPT
        }
    ],
)

# Print the translated Spanish text
print(response['message']['content'])
