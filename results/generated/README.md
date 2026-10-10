# Generated outputs

`make figures` writes PDF and PNG files to `figures/`. Those files are local rebuilds and are not part of the frozen release.

The immutable copies and SHA-256 manifest live in `../cached/`. Check them with:

```bash
make verify-cached
```
