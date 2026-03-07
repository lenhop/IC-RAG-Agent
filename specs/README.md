# API Specifications

This folder holds the canonical OpenAPI specification for the UDS Agent API.

## Files

- `UDS_API_SPEC.yaml` – Human-edited OpenAPI 3.1 spec (canonical source)

## Notes

- YAML is the single source of truth. If JSON is needed for tooling, generate it from the YAML (e.g. `openapi-generator` or `yq`).
- The live API schema is served at `/openapi.json` when the UDS API is running.
