from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db" / "chroma_db"

# Load GitHub Models API Key
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in .env file")

# OpenAI compatible client (GitHub Models)
client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN
)


# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load vector database
def load_vectorstore():
    print("Loading vector database...")

    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_model
    )

    return vectorstore


# Retrieve relevant documents with threshold filtering
def retrieve_documents(query, vectorstore, threshold=0.6):

    results = vectorstore.similarity_search_with_score(query, k=5)

    filtered_docs = []

    for doc, score in results:

        similarity = 1 - score

        if similarity >= threshold:
            filtered_docs.append((doc, similarity))

    # Sort by similarity (reranking)
    filtered_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in filtered_docs[:3]]


# Generate answer using GitHub LLM
def generate_answer(question, context, history):

    prompt = f"""
You are DocTalk AI, a healthcare assistant.

Your job is to explain medical information in simple language.

Rules:
- Only answer using the provided context
- If the answer is not in the context, say you do not know
- Never provide medical diagnosis
- Always add a short medical disclaimer

Conversation History:
{history}

Medical Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


# CLI chat loop
def main():

    print("DocTalk AI Health Assistant")

    vectorstore = load_vectorstore()

    conversation_history = ""

    while True:

        user_question = input("\nAsk a medical question (or type 'exit'): ")

        if user_question.lower() == "exit":
            print("Goodbye")
            break

        try:

            docs = retrieve_documents(user_question, vectorstore)

            if not docs:
                print("\nAnswer:\n")
                print("Sorry, I couldn't find relevant medical information for that question.")
                continue

            context = "\n\n".join([doc.page_content for doc in docs])

            answer = generate_answer(user_question, context, conversation_history)

            print("\nAnswer:\n")
            print(answer)

            # update conversation memory
            conversation_history += f"\nUser: {user_question}\nAssistant: {answer}\n"

        except Exception as e:

            print("\nError processing your request.")
            print(str(e))


if __name__ == "__main__":
    main()