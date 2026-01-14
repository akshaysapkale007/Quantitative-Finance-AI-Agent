# FinAI v1.0 — Legacy Archive

> ⚠️ **This code no longer runs** due to breaking changes in LangChain and OpenBB APIs.  
> It is preserved here to demonstrate the evolution of the project.

## Original Implementation (Early 2025)

This was the first version of the Finance AI Chatbot, built when:
- LangChain was still rapidly evolving (pre-v1.0)
- OpenBB required paid API keys
- Gemini 1.5 Flash was the latest model
- AI coding assistants were not yet integrated into IDEs

## Architecture

```
v1 Architecture:
┌─────────────────┐
│   Flask App     │
└────────┬────────┘
         │
┌────────▼────────┐
│  FinanceChatbot │
│     Class       │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────────┐
│Gemini │ │   Tools   │
│ 1.5   │ │  (paid)   │
└───────┘ └───────────┘
```

## Why It Broke

| Component | v1 (Broken) | v2 (Current) |
|-----------|-------------|--------------|
| LangChain imports | `from langchain.chat_models` | `from langchain_google_genai` |
| Agent creation | `create_tool_calling_agent` | `create_react_agent` |
| OpenBB auth | `obb.account.login(api_key=...)` | Free yfinance provider |
| Web search | Tavily API (paid) | Removed (not needed) |
| Model | `gemini-1.5-flash-latest` | `gemini-3-flash-preview` |

## Lessons Learned

1. **Dependency management** — Pin versions in production
2. **API abstraction** — Don't tightly couple to external APIs
3. **Free-tier alternatives** — yfinance provides same data without API keys
4. **Framework evolution** — LangChain's rapid changes required complete refactor

---

See the main [README.md](../README.md) for the current v2 implementation.
