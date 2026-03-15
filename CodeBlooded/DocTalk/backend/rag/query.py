from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "db" / "chroma_db"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN not found in .env file")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=GITHUB_TOKEN
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_vectorstore():
    print("Loading vector database...")
    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embedding_model
    )
    return vectorstore


def rewrite_query(question, history):
    prompt = f"""
Rewrite the user question into a standalone medical search query.

Conversation history:
{history}

User question:
{question}

Return ONLY the rewritten query.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You rewrite queries for search."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def retrieve_documents(query, vectorstore, threshold=1.2):
    results = vectorstore.similarity_search_with_score(query, k=6)

    filtered_docs = []

    for doc, score in results:
        if score <= threshold:
            filtered_docs.append((doc, score))

    filtered_docs.sort(key=lambda x: x[1])

    return [doc for doc, _ in filtered_docs[:4]]


def generate_answer(question, context, history):
    prompt = f"""
You are DocTalk AI, a healthcare assistant.

Use ONLY the provided medical context.

Rules:

- If the answer is not present in the context say:
  "I do not know based on the available medical data."
- Do not invent medical information
- Never provide diagnosis
- Explain medical information simply

Conversation History:
{history}

Medical Context:
{context}

User Question:
{question}

Provide a clear explanation.

Always end with:

Disclaimer: This information is for educational purposes only and not a substitute for professional medical advice.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a medical information assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content


def main():
    print("DocTalk AI Health Assistant")

    vectorstore = load_vectorstore()

    conversation_history = ""
    last_context = ""

    while True:
        user_question = input("\nAsk a medical question (or type 'exit'): ")

        if user_question.lower() == "exit":
            print("Goodbye")
            break

        try:
            search_query = rewrite_query(user_question, conversation_history)

            docs = retrieve_documents(search_query, vectorstore)

            if not docs and last_context:
                fallback_query = last_context + " " + user_question
                docs = retrieve_documents(fallback_query, vectorstore)

            if not docs:
                print("\nAnswer:\n")
                print("Sorry, I couldn't find relevant medical information for that question.")
                continue

            context = "\n\n".join([doc.page_content for doc in docs])

            last_context = context

            answer = generate_answer(user_question, context, conversation_history)

            print("\nAnswer:\n")
            print(answer)

            conversation_history += f"\nUser: {user_question}\nAssistant: {answer}\n"

        except Exception as e:
            print("\nError processing your request.")
            print(str(e))


if __name__ == "__main__":
    main()
