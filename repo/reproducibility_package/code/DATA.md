# Data artifacts

The online experiments read a single condensed `.npz` per configuration. Each
one is self-contained: context embeddings, action embeddings, labels, and the
three cached LLM proposals. **No GPU, no LLM, and none of the raw source data
are needed to reproduce any online result** — only the file for that
configuration.

These files are too large for a paper supplement, so they are distributed
separately. Verify what you download against these hashes:

| File | Size | SHA-256 |
|---|---|---|
| `amazon_v1_slim.npz` | 228 MB | `5a959961a2f6e38eb1a2d64aa8687b265d8407a334d54084213f5ed1887a2582` |
| `amazon_v3_slim.npz` | 303 MB | `306bcbae8cc4be3cdd20004a29ec64cf1b0071f0e6c03d85bc746a878fbd2f9c` |
| `wiki_v1_slim.npz` | 1.9 GB | `90d2555f17b01fa9855916c0b50d6d15388784ff7953a9766ead11e9107be254` |
| `wiki_v3_slim.npz` | 2.6 GB | `ef49cb5b49cb34945389267288dd75cb1c577f9b72c2c22ff996f079f6a0cc0f` |

Check a download with:

```bash
sha256sum amazon_v1_slim.npz      # or: shasum -a 256 on macOS
```

**Download location:** _to be filled in before release_ (an anonymized archive
link for review, a permanent one afterwards).

## Schema

Every slim file has the same keys, so one loader handles both datasets:

| Key | Shape | Meaning |
|---|---|---|
| `contexts` | `(n, d)` float32 | context embeddings |
| `Z` | `(K, d)` float32 | action embeddings |
| `y_flat` | `(nnz,)` int32 | correct action ids, concatenated across contexts |
| `y_offsets` | `(n+1,)` int64 | ragged offsets into `y_flat` |
| `n_keep` | scalar int64 | length of the online stream |
| `llm_small`, `llm_base`, `llm_large` | `(n,)` int64 | each expert's cached proposal |

`n` exceeds `n_keep` on AmazonCat (60,000 vs 50,000). The surplus is a held-out
pool used for two things that must never touch the online stream: the offline
arm's free expert pulls, and fitting the ridge oracle `theta_or`.

## Original sources

The raw datasets these files are derived from:

| Dataset | Source |
|---|---|
| OneShotWikiLinks | <https://www.kaggle.com/generall/oneshotwikilinks> |
| AmazonCat-13K | Extreme Classification Repository, <http://manikvarma.org/downloads/XC/XMLRepository.html> |

You only need these if you are rebuilding the embeddings or the LLM caches
from scratch. To reproduce any result in the paper, the condensed `.npz` files
above are sufficient.

## Rebuilding from source

Only needed if you want to change the embeddings or the experts.

1. `01_data_preparation/make_data.py` (BERT-768d) or `make_data_v3.py` (BGE-1024d)
   builds the full dataset pickle from the raw AmazonCat-13K release.
2. `02_llm_caching/cache_llm_predictions_amazon.py` (or `_wiki.py`) runs each
   LLM once over the dataset and caches its proposals. Decoding is greedy
   (`temperature=0`, `do_sample=False`), so the caches are deterministic.
3. `01_data_preparation/make_slim_npz.py` condenses the pickle plus the caches
   into the `.npz` above.

Steps 1 and 2 need a GPU and the optional dependencies in `requirements.txt`.
