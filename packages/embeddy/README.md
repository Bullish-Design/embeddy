# embeddy

Typed, metric-honest embedding, storage, search, and serving. Consumes chonkai.

- Keystone (M1): protocol types, EmbeddingProvider, model registry + MRL policy,
  Searchable (incl. source ops), sqlite-vec + FTS5 store, RRF/weighted fusion.
- Extras: `local`, `server`, `client`, `qdrant`. Core imports with zero extras.
