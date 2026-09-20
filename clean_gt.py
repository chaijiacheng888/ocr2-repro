"""Drop dangling reading-order edges from a released OmniDocBench GT file.

The released `OmniDocBench.json` carries an upstream data bug: on 12 pages, 37
`relation_type="truncated"` edges point at `anno_id`s that are absent from that page's
`layout_dets`, and the official scorer raises on the first one it walks.

Each page may carry edges of the form
`{"source_anno_id": 2, "target_anno_id": 3, "relation_type": "parent_son"}`. Those ids
must resolve to an `anno_id` on the same page, but the released GT mixes two id schemes
(`box_id_7` strings and bare ints), so some edges point at nothing. This keeps only the
edges whose endpoints resolve.

On the v1.6 GT this removes 131 edges (2852 -> 2721); 37 of them are `truncated` and the
other 94 are `parent_son`. The page-level scorer does not consume `parent_son`, so only
the `truncated` ones could crash it -- filtering both is simpler and equally correct.
No page is removed.

    python clean_gt.py OmniDocBench.json OmniDocBench_clean.json
"""

import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    dropped = 0
    for page in data:
        edges = (page.get("extra") or {}).get("relation") or []
        if not edges:
            continue
        ids = {
            value
            for det in (page.get("layout_dets") or [])
            for value in (det.get("anno_id"), str(det.get("anno_id")))
        }
        kept = [
            edge
            for edge in edges
            if edge.get("source_anno_id") in ids and edge.get("target_anno_id") in ids
        ]
        dropped += len(edges) - len(kept)
        page["extra"]["relation"] = kept

    with open(dst, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)

    print(f"pages: {len(data)}   dropped {dropped} dangling edges -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
