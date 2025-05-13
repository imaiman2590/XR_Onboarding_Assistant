
---

## 🧠 XR Onboarding Assistant (AI + LangChain + Hugging Face + Notion + React)

An intelligent, scalable onboarding assistant that answers HR and company policy questions using **retrieval-augmented generation (RAG)** with **Hugging Face LLMs**, internal docs, Notion databases, and web search. Accessible via a React UI and FastAPI backend, fully containerized with Docker.

---

### 🚀 Features

✅ Hugging Face LLM (`flan-t5-large`)
✅ RAG with FAISS and Notion documents
✅ Web search with DuckDuckGo
✅ React frontend UI
✅ FastAPI backend API
✅ Dockerized for easy deployment
✅ XR-ready structure (voice, immersive UI hooks)

---

## 🧩 Project Structure

```
xr-assistant/
├── backend/
│   ├── main.py                # FastAPI server
│   ├── agent.py               # LangChain agent setup
│   ├── qa_tools.py            # Document loading + FAISS
│   ├── knowledge/
│   │   └── onboarding_faq.txt # Onboarding Q&A
│   ├── .env                   # HuggingFace + Notion secrets
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                  # React app (your existing UI)
├── docker-compose.yml
└── README.md
```

---

## 📥 Installation

### 🔧 Requirements

* Docker and Docker Compose
* Hugging Face account and token
* Notion integration token and database ID (optional)

### 🛠️ Setup

1. **Clone the repo**

```bash
git clone https://github.com/your-org/xr-assistant.git
cd xr-assistant
```

2. **Configure environment variables in `backend/.env`**

```env
HUGGINGFACEHUB_API_TOKEN=your_hf_token
NOTION_INTEGRATION_TOKEN=your_notion_token
NOTION_DATABASE_ID=your_db_id
```

3. **Build & run the project**

```bash
docker-compose up --build
```

---

## 🌐 API Usage

### POST `/ask/`

Ask a question to the assistant.

#### Example Request:

```bash
curl -X POST http://localhost:8000/ask/ \
     -H "Content-Type: application/json" \
     -d '{"message": "What tools are provided to new hires?"}'
```

#### Example Response:

```json
{
  "response": "New hires receive a laptop, email credentials, VPN access, and access to Slack, Jira, Notion, and internal wikis."
}
```

---

## 🧪 Testing Locally

You can test locally using:

```bash
uvicorn main:app --reload --port 8000
```

Then hit: `http://localhost:8000/docs` for Swagger API testing.

---

## 📄 Onboarding FAQ Content

Update onboarding questions via:

```
backend/knowledge/onboarding_faq.txt
```

Each entry should follow this format:

```
Q: Your question here?
A: The answer here.
```

---

## 📦 Build Only Backend (Optional)

```bash
cd backend
docker build -t xr-assistant-backend .
docker run -p 8000:8000 xr-assistant-backend
```

---

## 🛡️ Security Notes

* Never commit `.env` with secrets.
* Use secrets management (e.g. AWS Secrets Manager) in production.
* Add CORS restrictions in `main.py` before deploying.

---

## 📌 Roadmap

* [ ] XR headset interaction (Unity/WebXR)
* [ ] Voice interface (Whisper + TTS)
* [ ] Admin panel to manage onboarding content
* [ ] Slack/Teams bot integration

---

## 🤝 Contributing

Pull requests and feature ideas are welcome. Please fork the repo and submit a PR.

---

## 📜 License

MIT License

---


