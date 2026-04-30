# Contributing

- Open an issue for scope changes.
- Add GEE datasets to `src/biodata/configs/ee_catalog.yml`; local datasets go via `update_catalog()` at runtime.
- Write tests for new reducers/adapters.
- Keep big data out of git; use `data/` for tiny samples only.
