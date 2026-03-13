⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/DedHarshit/MedTak.git
cd MedTak
2️⃣ Create virtual environment
python -m venv .venv

Activate it:

Windows:

.venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Add API key

Create a .env file inside the backend folder:

GITHUB_TOKEN=your_api_key_here
▶️ Running the Project
1️⃣ Build the vector database
python CodeBlooded/DocTalk/backend/rag/ingestion.py
2️⃣ Start the AI assistant
python CodeBlooded/DocTalk/backend/rag/query.py
