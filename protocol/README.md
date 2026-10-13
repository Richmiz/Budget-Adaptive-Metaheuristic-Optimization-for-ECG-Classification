# protocol/

Written only by `scripts/lock_protocol.py`:

- `frozen_protocol.yaml`: merged configuration snapshot with config hash, written at gate G2 (never overwritten)
- `analysis_lock.json`: hashes of the frozen protocol and docs/07_analysis_plan.md, written at gate G4. Fold 10 cannot be evaluated until it exists.
