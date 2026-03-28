from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

texts = [
    "Hello this is Tanishq Mathur",
    "Hello your name is VsCode",
    "And you all are very beautiful"
]

vector = embeddings.embed_documents(texts)

print(vector)
