# MakAgent — AI-Powered Campus Assistant

MakAgent is an AI-powered campus assistant I designed and developed to help Makerere University students and staff find clear, grounded answers from trusted institutional information.

This repository is a **sanitized public portfolio snapshot** created from my private development repository. It contains selected production-oriented source files that demonstrate the AI engineering work without exposing deployment secrets, credentials, internal infrastructure details, large source documents, or the private Git history.

## What MakAgent does

MakAgent combines retrieval, LLM generation, source validation, and an accessible chat interface to answer university-service questions in a safer and more useful way than a generic chatbot.

Key capabilities include:

- **Python + FastAPI** backend
- **Google Gemini** LLM integration
- **Retrieval-Augmented Generation (RAG)**
- **Sentence Transformers** embeddings
- **FAISS** vector retrieval
- Hybrid semantic + lexical retrieval
- Metadata-aware reranking
- Grounding and answer-safety checks
- Provider error classification and graceful fallbacks
- React frontend
- Voice input and spoken responses
- Accessibility-focused interaction
- Production deployment on Ubuntu/Proxmox behind Nginx

## Why I built it

Students often need help navigating services such as portals, university email, learning platforms, admissions, payments, accommodation, and ICT support. Generic LLMs may provide plausible but incorrect institutional answers.

MakAgent is designed to retrieve trusted university information first and then generate responses from approved evidence.

## High-level architecture

```text
User
  |
React chat + voice interface
  |
FastAPI API
  |
Query policy / routing
  |
RAG retrieval
  |-- Sentence Transformers embeddings
  |-- FAISS vector index
  |-- Keyword retrieval
  |-- Metadata-aware reranking
  |
Grounding / evidence validation
  |
Gemini generation
  |
Student-friendly response + verified source metadata
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for more detail.

## Selected code in this public snapshot

### Backend
- `backend/config.py` — environment configuration
- `backend/agent/gemini.py` — Gemini provider boundary and failure classification
- `backend/rag/embeddings.py` — offline local embedding runtime
- `backend/rag/faiss_store.py` — FAISS-backed document retrieval
- `backend/rag/reranking.py` — deterministic metadata-aware reranking

### Frontend
- `frontend/src/api/chatApi.js` — backend chat API client
- `frontend/src/hooks/useVoiceInput.js` — browser speech-recognition flow
- `frontend/src/hooks/useSpeechOutput.js` — spoken-answer flow

## Technologies

**AI / Retrieval:** Gemini, Sentence Transformers, FAISS, RAG  
**Backend:** Python, FastAPI  
**Frontend:** React, JavaScript  
**Infrastructure:** Ubuntu, Proxmox, Nginx  
**Engineering:** testing, source validation, failure handling, deployment troubleshooting

## Security

No live credentials are included in this repository. Environment files are represented only by safe placeholders in `.env.example`.

A Gemini credential that had previously existed in the private development repository was rotated and replaced before this public showcase repository was created. This public repository has **fresh Git history** and does not inherit the private repository's commit history.

## Project status

MakAgent has been deployed internally on Makerere University infrastructure and has been developed as an ongoing AI application project.

## My role

I designed and implemented the application across the AI pipeline, backend APIs, retrieval architecture, frontend interaction, voice/accessibility features, testing, deployment, and troubleshooting.

---

**Developer:** Barbra Gift Kyomuhendo  
**GitHub:** https://github.com/barbie256

> This public repository is a recruiter/portfolio showcase, not a deployment mirror of the private working repository.
