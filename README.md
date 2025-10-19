# 🤖 Feedback AI  
### Multi-Agent Ticket Management System (CrewAI + Groq + Streamlit)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-green)
![LLM](https://img.shields.io/badge/LLM-Groq%20LLaMA3.3--70B-purple)

---

## 🚀 Overview

**Feedback AI** is a local, privacy-friendly **multi-agent ticketing system** that classifies and summarizes customer feedback.  
It combines **CrewAI orchestration**, **Groq LLM reasoning**, and **Streamlit dashboards** to automate the triage of feedback into actionable tickets.

**Highlights:**
- 🧠 Multi-Agent orchestration: Classifier, Arbiter, Summarizer  
- ⚡ Hybrid reasoning: Heuristic + Groq LLM fallback  
- 📊 Real-time analytics and insights  
- 💾 Local persistence in CSV (no external DB)  
- 🧩 Knowledge base for pattern-aware classification  

---

## 🧩 Architecture

User Upload → Heuristic Classifier → (if uncertain) CrewAI Agents (Classifier + Arbiter)
↓
Summarizer Agent (CrewAI) → Markdown Summary
↓
Streamlit Dashboard → Tickets + Analytics


Agents used:

| Agent | Purpose |
|--------|----------|
| **Classifier** | Categorizes feedback into Bug, Feature, Complaint, or Praise |
| **Arbiter** | Validates and fixes classification JSON |
| **Summarizer** | Creates concise trend summaries |

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/feedback-ai.git
cd feedback-ai

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Environment Setup
cp .env.example .env

4️⃣ Run the App
streamlit run app.py


📊 Key Data Files
| File                         | Purpose                 |
| ---------------------------- | ----------------------- |
| `data/generated_tickets.csv` | All generated tickets   |
| `data/knowledge_base.csv`    | Knowledge base patterns |
| `data/config.json`           | App configuration       |


🧠 Knowledge Base
Define known feedback patterns manually in data/knowledge_base.csv.
Each row improves agent accuracy for repeated patterns.
| pattern     | category        | severity | priority | kb_notes            |
| ----------- | --------------- | -------- | -------- | ------------------- |
| login crash | Bug             | High     | High     | Known Android crash |
| dark mode   | Feature Request |          | Medium   | UI request          |

💡 Future Enhancements

 Ollama fallback for offline models

 Vector DB (FAISS / Chroma) for semantic KB

 SQLite backend

 Streamlit Cloud deployment

 API endpoints for external integrations


 🧑‍💻 Author
Tadepalli Siva Venkata Dheemanth
AI Engineer & Creator of Feedback AI

📜 License
This project is licensed under the MIT License
.