"""Run a page-image directory through a DeepSeek-OCR-2 SGLang server.

Produces one `.md` per page image, named after the image stem, which is the layout the
official OmniDocBench scorer consumes.

The request shape is the official one from `deepseek-ai/DeepSeek-OCR-2`, directory
`DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/` (`config.py` + `run_dpsk_ocr2_eval_batch.py`):
prompt `<image>\\n<|grounding|>Convert the document to markdown.`, temperature 0,
max_tokens 8192, and `NoRepeatNGramLogitsProcessor(ngram_size=40, window_size=90,
whitelist_token_ids={128821, 128822})`. That script drives vLLM in-process, so it cannot
talk to an SGLang server — this sends the same shape to `/generate` instead.

    python client.py --image-dir /data/images --output /data/pred_md --workers 12
"""

import argparse
import concurrent.futures as futures
import os
import re
import sys
import threading
import time

import requests

from sglang.srt.sampling.custom_logit_processor import (
    DeepseekOCRNoRepeatNGramLogitProcessor,
)

PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
ROUTE = "/generate"

_REF_DET_RE = re.compile(r"(<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>)", re.DOTALL)
_FORMULA_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)


def postprocess(text: str) -> str:
    """Official-style cleanup: strip grounding boxes, tidy formula LaTeX."""
    text = _REF_DET_RE.sub("", text)
    text = text.replace("\n\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

    def process(match):
        return r"\[" + re.sub(r"\\quad\s*\([^)]*\)", "", match.group(1)).strip() + r"\]"

    return _FORMULA_RE.sub(process, text)


def md_name(image_path: str) -> str:
    base = os.path.basename(image_path)
    for ext in (".jpg", ".jpeg", ".png"):
        if base.lower().endswith(ext):
            return base[: -len(ext)] + ".md"
    return os.path.splitext(base)[0] + ".md"


def build_payload(image: str, max_new_tokens: int) -> dict:
    return {
        "text": PROMPT,
        "image_data": image,
        "sampling_params": {
            "temperature": 0.0,  # greedy; SGLang maps temperature 0 to top_k 1
            "max_new_tokens": max_new_tokens,
            # keep <|ref|>/<|det|> literal so postprocess() can strip the grounding spans
            "skip_special_tokens": False,
            "custom_params": {
                "ngram_size": 40,
                "window_size": 90,
                "whitelist_token_ids": [128821, 128822],  # <td>, </td>
            },
        },
        "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="OCR a page directory into per-page markdown via a SGLang server.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--image-dir", required=True, help="dir of page images (jpg/jpeg/png)")
    ap.add_argument("--output", required=True, help="dir for per-page .md predictions")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="output ceiling; with --allow-auto-truncate the server clamps it to the "
        "remaining context",
    )
    ap.add_argument("--raw", action="store_true", help="skip postprocess()")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}{ROUTE}"
    images = sorted(
        name
        for name in os.listdir(args.image_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        ap.error(f"no jpg/jpeg/png images found in {args.image_dir}")
    os.makedirs(args.output, exist_ok=True)

    total = len(images)
    done = [0]
    failures = []
    lock = threading.Lock()
    started = time.time()

    def work(name: str) -> None:
        dst = os.path.join(args.output, md_name(name))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            with lock:  # resumable: a re-run skips pages already written
                done[0] += 1
            return
        session = requests.Session()
        try:
            resp = session.post(
                url,
                json=build_payload(os.path.join(args.image_dir, name), args.max_new_tokens),
                timeout=(30, 3600),
            )
            resp.raise_for_status()
            text = resp.json()["text"]
        except Exception as exc:  # one bad page must not discard the whole run
            with lock:
                failures.append((name, repr(exc)[:200]))
                done[0] += 1
                if len(failures) <= 10:
                    print(f"[{done[0]}/{total}] FAILED {name}: {exc}", file=sys.stderr)
            return
        finally:
            session.close()
        with open(dst, "w", encoding="utf-8") as handle:
            handle.write(text if args.raw else postprocess(text))
        with lock:
            done[0] += 1
            if done[0] % 25 == 0 or done[0] == total:
                rate = done[0] / max(time.time() - started, 1e-6)
                print(
                    f"[{done[0]}/{total}] {rate:.2f} pg/s  "
                    f"eta {(total - done[0]) / max(rate, 1e-6) / 60:.1f} min",
                    flush=True,
                )

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, images))

    print(f"done: {total - len(failures)}/{total} pages -> {args.output}")
    if failures:
        print(f"failed pages ({len(failures)}):", file=sys.stderr)
        for name, exc in failures[:40]:
            print(f"  {name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
