# Core Modules

Core business logic and integrations.

## Files

- **`config.py`** - Centralized configuration and environment variables ![Coverage: 86%](https://img.shields.io/badge/coverage-86%25-brightgreen)
- **`bedrock.py`** - AWS Bedrock model integration (Claude, Nova, Mistral) ![Coverage: 70%](https://img.shields.io/badge/coverage-70%25-yellow)
- **`rag.py`** - RAG system for document retrieval and context generation ![Coverage: 67%](https://img.shields.io/badge/coverage-67%25-yellow)
- **`vector_store.py`** - Pinecone vector database operations ![Coverage: 70%](https://img.shields.io/badge/coverage-70%25-yellow)
- **`initialize_rag.py`** - RAG system initialization script ![Coverage: 97%](https://img.shields.io/badge/coverage-97%25-brightgreen)

## Test Coverage Status

| Module | Coverage | Status |
|--------|----------|--------|
| `initialize_rag.py` | 97% | ✅ Excellent |
| `config.py` | 86% | ✅ Good |
| `bedrock.py` | 70% | ⚠️ Moderate |
| `vector_store.py` | 70% | ⚠️ Moderate |
| `rag.py` | 67% | ⚠️ Moderate |

*Coverage data from latest test run. Run `pytest --cov=src/core` for updated stats.*

## Usage

All modules use centralized configuration from `config.py`.