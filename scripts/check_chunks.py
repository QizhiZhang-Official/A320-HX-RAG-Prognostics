import collections
import json

chunks = []
with open("outputs/chunks.jsonl", "r", encoding="utf-8") as f:
    for ln in f:
        if ln.strip():
            chunks.append(json.loads(ln))

# 1) section 分布
dist = collections.Counter()
for c in chunks:
    dist[(c["manual_id"], c["section"])] += 1
for key in sorted(dist.keys()):
    print(key, dist[key])
print("total:", len(chunks))

# 2) 不应有 TOC/front matter 污染
bad = 0
for c in chunks:
    if "NOT APPLICABLE" in c["text"] or "...." in c["text"]:
        bad += 1
print("toc-contaminated:", bad)

# 3) 不应有 IPL 内容混入
ipl = 0
for c in chunks:
    if "NOMENCLATURE" in c["text"] or "ITEM NOT ILLUSTRATED" in c["text"]:
        ipl += 1
print("ipl-contaminated:", ipl)

# 4) 关键内容在位
hit_sigma = 0
hit_ten = 0
for c in chunks:
    if "σΔP" in c["text"]:
        hit_sigma += 1
    if "TEN PER CENT" in c["text"]:
        hit_ten += 1
print("chunks with sigma-dP:", hit_sigma)
print("chunks with TEN PER CENT:", hit_ten)

# 5) 21-52-12 的 SPECIAL TOOLS 页必须归类正确
for c in chunks:
    if c["manual_id"] == "CMM-21-52-12" and c["manual_page_start"] >= 9000:
        print(c["chunk_id"], c["section"], c["manual_page_start"], "-", c["manual_page_end"])