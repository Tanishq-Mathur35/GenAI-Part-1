from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model = "mistral-small-2506", temperature=0.9, max_tokens=20)

response = model.invoke("Write a Poem on AI?")

print(response.content)



# from langchain_groq import ChatGroq

# response2 = model.invoke("Give me a para on ML?")

# model2 = ChatGroq(model="openai/gpt-oss-120b")

# print(response2.content)
