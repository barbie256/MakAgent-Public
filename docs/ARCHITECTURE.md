# MakAgent Architecture

## Request flow

1. A user asks a question through the React chat interface.
2. FastAPI receives the request.
3. Query policy separates public guidance from requests that require authenticated personal data.
4. The retrieval pipeline searches trusted institutional knowledge.
5. Sentence-transformer embeddings and FAISS provide semantic retrieval.
6. Keyword matching and metadata-aware reranking improve exact matches for service names, acronyms, and university terminology.
7. Grounding logic checks that retrieved evidence is relevant and suitable for the requested response type.
8. Approved evidence is passed to Gemini with constrained instructions.
9. The frontend presents the answer together with trust/source metadata and optional spoken output.

## Retrieval

MakAgent combines:

- semantic vector similarity
- keyword matching
- title relevance
- phrase matching
- service-entity matching
- response-intent compatibility
- domain metadata
- source authority
- evidence completeness

This hybrid approach is useful for university environments where exact service names such as portal names and acronyms matter.

## Reliability

The LLM integration is isolated behind a provider boundary that classifies rate limits, authentication problems, provider unavailability, timeouts, malformed responses, and unavailable models.

The application can return a safe fallback instead of leaking provider details to users.

## Accessibility

The React interface includes browser-based voice input and speech synthesis, keyboard-friendly controls, live status announcements, and accessibility-focused interaction patterns.

## Deployment

The private working deployment runs on Ubuntu/Proxmox with Nginx in front of the application. Deployment-specific secrets, host information, and operational configuration are intentionally excluded from this public portfolio repository.
