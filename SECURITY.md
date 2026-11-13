# Security and data handling

No credentials or user interaction exports are distributed. Configure secrets only in the local process environment or an ignored `.env` file. The application does not automatically load `.env`; export variables before starting it. Do not commit learned embeddings, user identifiers, MongoDB dumps, source URLs with embedded authentication, or logs. The committed benchmark contains only aggregate synthetic results.

The legacy HTTP adapter defaults to loopback and requires a configured bearer key for POST operations. It is not an internet-facing multi-tenant service. Report a suspected exposure privately through the repository owner's GitHub profile; do not put the credential in an issue.
