import os
from dotenv import load_dotenv
from google import genai

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# O SDK inicializa buscando automaticamente a variável GEMINI_API_KEY
client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Explique o que é uma variável de ambiente em uma frase.",
)

print(response.text)