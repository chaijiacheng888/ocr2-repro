# DeepSeek-OCR-2 768px geometry — OmniDocBench reproduction

Reproduction steps for **sgl-project/sglang#38996** (*Serve DeepSeek-OCR-2 with its official
768px local-crop geometry*), run against a SGLang server built from that branch.

**Conventions.** Anything not in this repo gets an authoritative source plus a command.
Anything that has to be changed is written out explicitly. Anything already written in the
referenced files is not repeated here.

Contents:

| file | purpose | used in |
| --- | --- | --- |
| `client.py` | drives a SGLang server over `/generate` into per-page markdown | step 6 |
| `clean_gt.py` | drops the dangling reading-order edges the released v1.6 GT carries | step 4 |

---

## 1. SGLang, from the PR branch

| | |
| --- | --- |
| repository | `https://github.com/chaijiacheng888/sglang` |
| branch | `ds-ocr2-768` |
| **commit used for these numbers** | **`34c4711f5e516032297b50eec07a860044786c54`** |
| upstream base | `sgl-project/sglang:main`, PR base `33d46376a64d` |

```bash
git clone -b ds-ocr2-768 https://github.com/chaijiacheng888/sglang.git
cd sglang
git checkout 34c4711f5e516032297b50eec07a860044786c54
pip install -e python/
```

The branch changes four pure-Python files (preprocessing and token assembly) and no
kernels, so no `sglang-kernel` build is needed and no particular CUDA version is required.

Platform differences show up only in the server's launch flags; the batch client is
identical on both. Every command below works on CUDA and on Ascend.

## 2. Model weights

Source: `deepseek-ai/DeepSeek-OCR-2`.

```bash
huggingface-cli download deepseek-ai/DeepSeek-OCR-2 --local-dir /path/to/DeepSeek-OCR-2

# if HuggingFace is unreachable
modelscope download --model deepseek-ai/DeepSeek-OCR-2 --local-dir /path/to/DeepSeek-OCR-2
```

## 3. OmniDocBench

Source: `opendatalab/OmniDocBench`.

**v1.6 uses `main`; v1.5 uses the `v1_5` branch.** The two datasets are not
interchangeable — v1.6 re-annotated the same 1355 pages, so it is not a superset.

```bash
git clone https://github.com/opendatalab/OmniDocBench.git                 # v1.6
git clone -b v1_5 https://github.com/opendatalab/OmniDocBench.git odb_v15 # v1.5
```

## 4. Clean the ground truth

Required. The released `OmniDocBench.json` has 12 pages carrying 37 dangling
`relation_type="truncated"` reading-order edges whose endpoints are absent from that page's
`layout_dets`; the official scorer raises on the first one it walks.

```bash
python clean_gt.py OmniDocBench.json OmniDocBench_v16_clean.json
# pages: 1651   dropped 131 dangling edges -> OmniDocBench_v16_clean.json
```

131 edges go (37 `truncated` + 94 `parent_son`); the page count is unchanged. Only the
reading-order metric on those 12 pages is affected.

## 5. Start the server

```bash
python -m sglang.launch_server \
    --model-path /path/to/DeepSeek-OCR-2 \
    --enable-multimodal \
    --context-length 8192 --allow-auto-truncate --enable-custom-logit-processor
```

CUDA and Ascend use this same command — the platform-specific flags are defaults SGLang
resolves itself:

| flag | why it is not needed |
| --- | --- |
| `--device npu` | `device` defaults to `None`; `get_device()` auto-detects |
| `--attention-backend ascend` | Ascend platform default (`hardware_backend/npu/utils.py`) |
| `--page-size 128` | Ascend platform default |
| `--mm-attention-backend ascend_attn` | `layers/attention/vision.py`: `elif _is_npu: backend = "ascend_attn"` |
| `--disable-cuda-graph`, `--skip-server-warmup` | optional; they affect startup only |

**`--allow-auto-truncate` is not optional.** A 768px page expands to about 1100 image
tokens; asking for 8192 output tokens on top of that exceeds the 8192 window. With the flag
the server clamps the output budget to `8192 - input`; without it the request is rejected
outright.

### Check that 768px is actually in effect

```bash
curl -s http://127.0.0.1:30000/generate -H 'Content-Type: application/json' -d '{
  "text": "<image>\n<|grounding|>Convert the document to markdown.",
  "image_data": "/path/to/images/<any-page>.png",
  "sampling_params": {"max_new_tokens": 1, "skip_special_tokens": false}
}' | python -c "import json,sys; print(json.load(sys.stdin)['meta_info']['prompt_tokens'])"
```

| result | meaning |
| --- | --- |
| **1131** | 768px crops: 6 x 144 + 256 global + 1 separator + 9 text + 1 `<image>` |
| **867** | 640px crops — the change is not in effect |

This reads the token count the server actually produced, so it cannot be fooled by reading
the source.

## 6. Pages to per-page markdown

The scorer consumes one `.md` per page, named after the page image stem.

```bash
python client.py --image-dir /path/to/images --output /path/to/pred_md --workers 12
```

Any client will do; this one sends the official request shape (prompt
`<image>\n<|grounding|>Convert the document to markdown.`, temperature 0, max_tokens 8192,
`NoRepeatNGram(ngram_size=40, window_size=90, whitelist_token_ids={128821, 128822})`),
strips the grounding spans, and resumes over pages already written.

## 7. Score

Use OmniDocBench's **own** `configs/end2end.yaml` and repoint exactly two paths:

- `dataset.ground_truth.data_path` -> the cleaned v1.6 GT from step 4
- `dataset.prediction.data_path` -> the markdown directory from step 6

**Leave every other field alone.** The metric set, `match_workers`, `cdm_workers` and the
four timeout values are the shipped defaults, and the timeouts are load-bearing: they decide
whether a slow page falls back to chunked Hungarian matching, which changes the
`reading_order` score.

The edited config has to sit in the mounted data directory — the image ships its own copy at
`/workspace/configs/`, which is not the same place as `/data`.

```bash
cp /path/to/OmniDocBench/configs/end2end.yaml /path/to/data/end2end.yaml
# ...edit the two paths above, then:

docker run --rm \
    -v /path/to/data:/data \
    -v /path/to/out:/workspace/result \
    ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204 \
    --config /data/end2end.yaml
```

The image pins the TeX Live / ImageMagick / Ghostscript toolchain the CDM metric needs.
Upstream `opendatalab/OmniDocBench` at `main` works too.

**v1.5** is the same shape with two differences: `ground_truth` points at the v1.5 dataset,
and `display_formula` uses `CDM_plain` instead of `CDM`. The v1.5 CDM shells out to `node`
for its LaTeX tokenizer and runs as two steps — export the matched formula pairs, then
score them with `metrics/cdm/evaluation.py`.

## 8. Reading the results

```
out/result/<prediction>_quick_match_run_summary.json   -> notebook_metric_summary   <- report this
out/result/<prediction>_quick_match_metric_result.json -> *.all.*                   <- not this
```

The same metric appears under two aggregations:

| path | meaning |
| --- | --- |
| `<metric>.all.<sub>.all` | unweighted per-sample mean |
| `<metric>.page.<sub>.ALL` | the leaderboard-style figure used in the tables |

`notebook_metric_summary.overall_notebook` is written by the scorer itself:

```
Overall = ((1 - text_edit) * 100 + CDM * 100 + TEDS * 100) / 3
```

## 9. Reference numbers

v1.6, 1651 pages:

| | Overall | text Edit | TEDS | TEDS-S | CDM | Read Order |
| --- | --- | --- | --- | --- | --- | --- |
| RTX 4090 — 640 | 89.76 | 0.0892 | 84.26 | 88.61 | 93.94 | 0.1619 |
| **RTX 4090 — 768** | **91.75** | **0.0519** | 86.39 | 89.78 | 94.06 | 0.1430 |
| Ascend 910B — 640 | 89.10 | 0.0985 | 84.03 | 88.00 | 93.11 | 0.1715 |
| **Ascend 910B — 768** | **91.95** | **0.0533** | 86.70 | 90.17 | 94.48 | 0.1444 |

v1.5, 1355 pages (Ascend): 640 -> 87.91, 768 -> 91.17.

Expected variation: floating-point non-determinism across hardware and backends moves
`text Edit` by ~0.001 and `Overall` by ~0.1. The direction (768 better on every metric)
and the magnitude (`text Edit` roughly halving) should reproduce consistently.

---

## When something does not work

| symptom | cause |
| --- | --- |
| scorer raises | step 4 was skipped; the 37 dangling edges are still there |
| scorer finds no matching predictions | the `.md` names are not the image stems, or the count is wrong |
| request rejected for exceeding context | `--allow-auto-truncate` is missing |
| `prompt_tokens` is 867 rather than 1131 | the 768px geometry is not in effect |
| scores far lower than the tables | `temperature` was not 0, or `max_tokens` truncated the pages |
| `reading_order` differs a lot | the `timeout_fallback_*` defaults were changed in step 7 |
| numbers ~2 points below the tables | the wrong field was read in step 8 (`*.all.*` instead of `.page.*.ALL`) |
