# Protected state for one-shot Codex prompt acquisition

Status: accepted and implemented in 1.32.0, 2026-10-30. Native gates and the separately staged whole-command paired-mean gate passed; [acceptance evidence and limits](../../specs/2026-10-30-codex-prompt-state-cache/context.md#production-acceptance) retain the measured scope.

The development prototype established exact new-prompt parity with a substantial native acquisition gain, but its state disappeared when the process exited. The production consumer invokes separate one-shot commands. Replaying the whole source to authenticate every checkpoint/ledger would discard that gain; plaintext or publicly checksummed state cannot certify the complete semantic projection merely because its source digest matches.

Retain the complete bounded proven engine state in a dedicated Windows current-user DPAPI store and make reuse explicit through the existing candidate command. Combine OS at-rest protection with an inner integrity envelope, current parser-image/profile/path binding and fresh authoritative source/lineage proof. Use a bounded nonblocking OS directory lock and ciphertext-only atomic publication. Cache failure remains an ordinary complete read. The [owning specification](../../specs/2026-10-30-codex-prompt-state-cache/spec.md) defines the exact contract, limits, override, threat scope and tests.

A retained worker would avoid persistent plaintext as well, but introduces endpoint authentication, process lifetime, recovery and ownership mechanisms. It is deferred to keep the selected slice independent of Companion, services and Notifier. A metadata-only cache still contains influential parser/checkpoint/reference state and does not remove the authenticity requirement. DPAPI is not executable attestation against malicious same-user processes; that stronger host-compromise boundary is not claimed. Unsupported platforms preserve the full read path.

This capability does not remove the cold full seed or establish end-to-end latency by itself. Whole-ccmsg acquisition and exact proof/refusal parity remain release acceptance gates, separately from the prototype's native component measurements.
