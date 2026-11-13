# Internal engineering guide

## Two execution paths

`recommender_lab.py` is the portable evaluation path. It is independent of the original HTTP/framework adapters, so its measured results must never be attributed to those adapters. The legacy path is `app.py` → `fit.py` / `recom.py` / `popular.py` → `classes/`.

`Core` stores configuration, the current request, a model instance, and status. The HTTP lock protects that shared request/model state within one process. It does not coordinate multiple server processes or make training durable.

## Adapter selection and request fields

`fit.py` lazily selects one model using the integer `model` field: 1=TFRS, 2=fastai, 3=LightFM, 4=GMF, 5=NCF, 6=custom collaborative filtering, 7=NNHS. `days` sets the requested interaction-history window. Read the chosen class's `fit` and data preparation methods before connecting a real source: framework-specific semantics and source schemas have not been unified.

`recom.py` reads `user_id` and an optional integer `top` (1–100), delegates to the model's `predict`, and truncates the result. `popular.py` returns that model's `popular_list`. Some historical classes support additional request/configuration details; there is no claim of a universal cross-framework API beyond these dispatch operations.

## Historical implementation notes

The integrations preserve several different learning approaches: retrieval with learned embeddings, neural/generalized collaborative filtering, LightFM interaction factorization, fastai embedding regression, and a custom scoring path. Their import dependencies are loaded only when the corresponding model is selected.

Old commented experiments and local troubleshooting instructions were replaced by this guide. The preserved classes still contain research limitations: certain metrics come from training history, some preprocessing/splitting needs a leakage review, and time-window handling contains historical assumptions. Do not present their printed metrics as held-out evidence. The optional dependency inventory intentionally does not promise that every framework can coexist in one interpreter.

## Portable evaluation contract

Events are globally sorted by timestamp. Train/validation/test boundaries keep equal timestamps together. Item and user vocabularies are learned from training data only. BPR negative sampling uses the training catalog and training positives; future labels do not influence training negatives. Recommendations exclude training positives. Unknown users fall back to popularity; the headline evaluator separately excludes cold users and unknown items and reports those exclusions.

The synthetic generator creates distinct user–item pairs and group preferences. Its simplicity explains why a small factor model can improve on global popularity. Real evaluation should add repeated-event policy, session handling, candidate coverage, popularity bias, confidence intervals, and business-specific relevance definitions before any quality claim.

## Operation and maintenance

1. Keep source credentials in an ignored local environment file or process environment.
2. Use a dedicated environment for a selected legacy adapter; record exact package versions after validation.
3. Run the offline checks before a commit.
4. Save model artifacts outside Git and use explicit model versions in deployment.
5. Never log authorization headers, connection strings, or individual interaction histories.

The service binds to loopback by default. A missing API key disables POST operations; a missing trained model returns HTTP 409. Training, recommendation, and popularity calls are serialized. External source requests have a timeout, but pagination semantics and operational retry policies need validation against the actual approved source.
