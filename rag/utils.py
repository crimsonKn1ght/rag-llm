import json
import math

def format_docs(docs):
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

def cosine(a, b):
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x*y for x,y in zip(a,b)) / (na * nb)

def get_unique_union(document_lists):
    seen = set()
    out = []
    for sublist in document_lists:
        for d in sublist:
            key = (
                getattr(d, "page_content", ""),
                json.dumps(getattr(d, "metadata", {}), sort_keys=True)
            )
            if key not in seen:
                seen.add(key)
                out.append(d)
    return out
