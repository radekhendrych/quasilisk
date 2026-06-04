# Agent Notes

- Keep this repo dependency-light unless there is a clear reason to add a package.
- Use `python3 -m unittest discover -s tests` after changes. For objective, seed, config, or runner changes, also run the relevant generic dry-run config, such as `python3 quasilisk.py --config configs/dryrun_censorship.json`.
- Do not limit yourself to content-related perturbations - stylistic/spelling and other perturbations are all admissible should they align with the stated objective.
- Preserve `events.jsonl` as the complete audit log of target inputs and responses.
