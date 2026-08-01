"""Fixed eval corpus for the retrieval-quality harness (CONCEPT §9.6).

~20 docs, 8 queries, qrels. Topic: the fictional "acme" developer platform.
Keywords are deliberately distinctive so the deterministic bag-of-words
FakeProvider yields non-trivial (not 0.0, not 1.0) metrics — the harness
must be able to DETECT retrieval regressions, which requires signal.

Ported VERBATIM from the Phase-0 spike (spikes/eval/corpus.py) — the gate
thresholds are derived from THIS exact corpus, so it must not drift.
"""

from __future__ import annotations

CORPUS: dict[str, str] = {
    # --- authentication -------------------------------------------------
    "auth-overview": (
        "Authentication overview. All acme API requests require an API key "
        "passed in the Authorization header. Keys start with the prefix "
        "acme_live_ and can be created in the dashboard under API Keys."
    ),
    "auth-scopes": (
        "API key scopes. Scopes restrict what a key can do: read, write, "
        "admin. A read-only key cannot create webhooks or modify billing "
        "settings. Review scopes before rotating keys."
    ),
    "auth-oauth": (
        "OAuth 2.0 for the acme platform. Redirect-based login flow returns "
        "an access token with a one-hour expiry and a refresh token. "
        "Refresh tokens rotate on every use."
    ),
    # --- billing ---------------------------------------------------------
    "billing-plans": (
        "Billing plans: Free, Pro, and Enterprise. Pro includes unlimited "
        "webhooks and 1000 API requests per minute. Enterprise adds SSO and "
        "a dedicated support SLA. Invoices are generated monthly."
    ),
    "billing-invoices": (
        "Invoices and receipts. Invoices appear in the dashboard on the "
        "first day of each month. Download past invoices as PDF or CSV. "
        "Payment method defaults to the primary card on file."
    ),
    # --- rate limits -----------------------------------------------------
    "ratelimits": (
        "API rate limits. Free plan allows 60 requests per minute. Pro and "
        "Enterprise allow 1000 requests per minute. Exceeding the limit "
        "returns HTTP 429 with a Retry-After header."
    ),
    "ratelimit-backoff": (
        "Handling 429 responses. Implement exponential backoff with jitter. "
        "Respect the Retry-After header. Retry at most three times before "
        "surfacing the error to the caller."
    ),
    # --- python client ---------------------------------------------------
    "client-python": (
        "Python client library. Install with pip install acme-python. "
        "Create a client with Acme(api_key=...). Methods map 1:1 to the "
        "REST API: resources.list(), resources.get(id)."
    ),
    "client-errors": (
        "Error handling in the Python client. Exceptions mirror HTTP status "
        "codes: RateLimitError, AuthError, ValidationError. Catch the base "
        "AcmeError to handle all failures uniformly."
    ),
    # --- webhooks --------------------------------------------------------
    "webhooks": (
        "Webhooks notify your server of events: charge.succeeded, "
        "invoice.paid, source.deleted. Endpoints must respond 200 within "
        "ten seconds. Failed deliveries retry with exponential backoff."
    ),
    "webhook-signatures": (
        "Verifying webhook signatures. Each delivery includes an "
        "X-Acme-Signature header computed with HMAC-SHA256 over the raw "
        "body. Rotate the signing secret via the dashboard."
    ),
    # --- search ----------------------------------------------------------
    "search-basics": (
        "Search API basics. The search endpoint accepts a query string and "
        "returns ranked results with relevance scores. Results include the "
        "matched snippet and a document id."
    ),
    "search-ranking": (
        "Search ranking and filters. Boost results by recency or by custom "
        "metadata. Filter results by content type, source path prefix, and "
        "chunk type before ranking."
    ),
    # --- errors ----------------------------------------------------------
    "error-codes": (
        "Error codes reference. 400 validation_error, 401 auth_error, "
        "403 permission_denied, 404 not_found, 409 conflict, 429 "
        "rate_limited, 500 internal_error."
    ),
    # --- deployment ------------------------------------------------------
    "deploy-regions": (
        "Deployment regions. acme runs in us-east-1, eu-west-1, and "
        "ap-southeast-1. Choose the region closest to your users. Data is "
        "not replicated across regions."
    ),
    "deploy-environments": (
        "Environments: development, staging, production. Each environment "
        "has its own API keys and webhook signing secrets. Promote between "
        "environments via the CLI."
    ),
    # --- data export -----------------------------------------------------
    "export-overview": (
        "Data export. Export your charges, customers, and subscriptions as "
        "CSV or JSON. Exports are generated asynchronously; download the "
        "result from a signed URL that expires after 24 hours."
    ),
    # --- dashboard -------------------------------------------------------
    "dashboard-overview": (
        "Dashboard guide. The dashboard shows live request metrics, error "
        "rates, and latency percentiles. The activity log records API calls "
        "with the key used and the response status."
    ),
    # --- api versioning --------------------------------------------------
    "versioning": (
        "API versioning. The API version is set per account. Versions are "
        "date-stamped, e.g. 2026-07-01. Breaking changes ship in a new "
        "version; old versions stay available for 12 months."
    ),
    # --- sso -------------------------------------------------------------
    "sso-saml": (
        "Single sign-on with SAML. Enterprise plans can configure SAML via "
        "the dashboard. After setup, team members sign in through your "
        "identity provider instead of passwords."
    ),
}

QUERIES: dict[str, str] = {
    "q1": "how do I authenticate API requests with an API key",
    "q2": "what scopes can I assign to an API key",
    "q3": "how much does the pro plan cost and what is included",
    "q4": "how do I handle HTTP 429 rate limit errors",
    "q5": "install and use the python client library",
    "q6": "verify webhook signatures",
    "q7": "how does search ranking and filtering work",
    "q8": "what do the error codes mean",
}

# qid -> set of relevant doc ids
QRELS: dict[str, set[str]] = {
    "q1": {"auth-overview", "auth-scopes"},
    "q2": {"auth-scopes"},
    "q3": {"billing-plans"},
    "q4": {"ratelimits", "ratelimit-backoff"},
    "q5": {"client-python", "client-errors"},
    "q6": {"webhook-signatures"},
    "q7": {"search-basics", "search-ranking"},
    "q8": {"error-codes"},
}
