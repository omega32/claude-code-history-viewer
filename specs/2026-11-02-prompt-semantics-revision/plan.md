# Implementation plan

1. Add a failing capability wire regression for the explicit initial epoch and feature, preserving existing command/feature expectations.
2. Declare the maintained epoch in shared prompt-candidate code and expose only the additive capability field/feature.
3. Document bump ownership and scope, honest legacy/null and malformed-advertisement behavior, and separation from source-bound revisions and retained-state compatibility versions.
4. Synchronize one additive native minor from 1.33.0 to 1.34.0, obtain independent review, and run focused then mandatory full isolated Rust, isolated/native all-target/all-feature Clippy and formatting gates serially.
5. Coordinate frontend generation at an idle boundary and build the configured native release only after the coordinator's explicit readiness/gate signal. The coordinator owns composed cleanup revision, reader validation and acceptance; no Notifier deployment or configuration change belongs here.

## Completed boundary

The maintained native epoch, additive feature/field, bounded wire regression, maintenance guidance and exact plan tracking are implemented. The one 1.34.0 minor synchronization, required full/static gates, independent source review, coordinated frontend and configured release build passed. Actual capabilities report API 1 and string epoch `"1"`; the coordinator verified legacy/null and declared-revision reader paths, private content/identity parity, conditional behavior, semantic-versus-source revision separation and current Notifier reader compatibility. See [validation and producer acceptance](context.md#producer-acceptance) for scope and retained evidence.

Native normalization, candidate arrays, snapshot/cache compatibility versions and Notifier runtime remain unchanged. Future Notifier cache adoption and the independent unavailable-task diagnosis are outside this completed producer slice.
