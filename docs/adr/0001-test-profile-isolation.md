# 0001: Compile-time test profiles with scoped fixture state

Status: Accepted

## Context

Windows KnownFolder resolution does not follow HOME/USERPROFILE. The previous opt-in unit shim protected repaired modules but left native lookups elsewhere and did not affect the production library linked into integration tests. Its shared global home also leaked between test threads. [The owning specification](../../specs/2026-10-08-test-profile-isolation/spec.md) records the incident and follow-up requirements.

## Decision

One Rust adapter owns native directory and profile-environment resolution. Rename the directory dependency and deny direct native methods through Clippy, supplemented by the supported runner's source preflight. Unit compilation and an explicit build-script activation enable fail-closed fixture resolution. Scoped state is thread-local, restored during unwinding, and explicitly propagated to owned blocking workers. Integration tests acquire a fixture scope before work.

The runner selects compile-time `CCHV_TEST_PROFILE=isolated-v1` and a separate Cargo target directory. A different explicit `live-v1` activation permits ignored live diagnostics; the supported live command additionally requires an acknowledgement argument. Missing or invalid activation cannot silently become live test access. Ordinary application builds retain native paths, including debug and all-feature builds. Runtime environment activation and an additive Cargo feature were rejected because they could change a deployed app's path behavior or silently activate through `--all-features`.

## Consequences

Default tests cannot inherit application profile paths through this adapter, and plain unactivated integration tests fail before profile access. Unpropagated threads fail closed, so new worker paths must use the shared propagation helper. Separate test targets cost additional initial compilation/storage but avoid replacing normal application artifacts. Local Cargo/Rustup configuration and caches remain trusted build inputs; the runner rejects inherited command/compiler overrides. This is a reviewed application boundary rather than an OS filesystem sandbox; explicit arbitrary paths, new native APIs and subprocesses remain review responsibilities.
