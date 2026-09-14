# Weight provenance and DIMER hosting

- Upstream: `microsoft/table-transformer-structure-recognition-v1.1-all`
- Immutable revision: `7587a7ef111d9dcbf8ac695f1376ab7014340a0c`
- Weight format: SafeTensors (`model.safetensors`, 115,437,156 bytes); the upstream repository hosts no pickle checkpoint at this revision.
- Manifest: `weights/table-transformer-structure-v1.1-all/dimer-base-manifest.json` (4 files: `README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json`; 115,515,347 bytes total, per-file SHA-256)
- Upstream weight license: MIT (the checkpoint's `README.md` front matter)
- DIMER hosting: MIT permits use, modification, distribution, and commercial use subject to preservation of the copyright notice and licence text. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `TableTransformerForObjectDetection` / `AutoImageProcessor` with `trust_remote_code=False`, `local_files_only=True` from the verified directory; the backbone is the in-library ResNet described by the config's `backbone_config` (`use_timm_backbone: false`), so nothing is fetched at construction (the smoke run loaded with `HF_HUB_OFFLINE=1`). The image-processor `size` is passed as `{"shortest_edge": 800, "longest_edge": 800}` because the pinned `preprocessor_config.json` form (`{"longest_edge": 800}`) is refused by the pinned transformers release; the two are the same resize.
