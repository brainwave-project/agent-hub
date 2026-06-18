# proto/

`knowhub.proto` is copied here from `know-hub/proto/` (same convention as
`authhub.proto`) and the Python stub is generated at build time via
`grpcio-tools`.

**Status:** deferred to M6. CRAG (M2–M5) retrieves over the existing know-hub
HTTP `/api/semantic` path; the gRPC write contract (`PersistChunks` etc.) is
only introduced once the ingestion crew's write path needs it — and only if a
measured payload/latency problem justifies it over plain HTTP/JSON
(review Issue 2, deferring ADR-AGT-08).
