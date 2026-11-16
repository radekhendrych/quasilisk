# Quasilisk Observatory Phase 0 prototype

This is a static, nonbinding interaction prototype used to review information
architecture, visual direction, terminology, and core workflows before the
frontend contract is added to the binding redesign specification.

The corresponding design proposal and six-pass critique record are in
`../quasilisk_frontend_phase_0_design.md`.

It uses representative data and performs no model calls, file writes, or
scientific reduction. Open `index.html` through a local static HTTP server so
the navigation and wizard interactions can be reviewed.

```bash
python3 -m http.server 8123 --directory docs/frontend_prototype
```

Then visit `http://127.0.0.1:8123/`.

Useful prototype routes:

- `#library`
- `#new-run/1` through `#new-run/4`
- `#run/overview`
- `#run/atlas`
- `#run/trace`
- `#run/strategy`
- `#run/operations`
- `#run/audit`

Static review captures are available as `preview-overview.png` and
`preview-configure.png` in this directory.
