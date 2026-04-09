import os
import json
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ----------------------------
# CONFIG
# ----------------------------
client = OpenAI(
    base_url="YOUR_INTERNAL_OPENAI_URL",
    api_key="YOUR_API_KEY"
)

MODEL = "gpt-4o-mini"


# ----------------------------
# LOAD CORPUS
# ----------------------------
def load_corpus():
    docs = []

    for file in os.listdir("corpus"):
        if file.endswith(".json"):
            with open(f"corpus/{file}") as f:
                docs.append(json.load(f))

    return docs


# ----------------------------
# NODE
# ----------------------------
class Node:
    def __init__(self, title, summary, text, source):
        self.title = title
        self.summary = summary
        self.text = text
        self.source = source
        self.children = []


def build_nodes(doc):
    nodes = []

    for sec in doc["toc"]:
        node = Node(sec["title"], sec["summary"], doc["text"], doc["source"])

        for sub in sec.get("subsections", []):
            child = Node(sub["title"], sub["summary"], doc["text"], doc["source"])
            node.children.append(child)

        nodes.append(node)

    return nodes


# ----------------------------
# BUILD BM25 INDEX
# ----------------------------
def build_bm25(nodes):
    corpus = []
    mapping = []

    for node in nodes:
        text = f"{node.title} {node.summary}"
        tokens = text.lower().split()

        corpus.append(tokens)
        mapping.append(node)

        for child in node.children:
            text = f"{child.title} {child.summary}"
            tokens = text.lower().split()

            corpus.append(tokens)
            mapping.append(child)

    bm25 = BM25Okapi(corpus)

    return bm25, mapping


# ----------------------------
# FAST RETRIEVAL (BM25)
# ----------------------------
def bm25_search(query, bm25, mapping, top_k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    top_idx = np.argsort(scores)[::-1][:top_k]

    return [mapping[i] for i in top_idx]


# ----------------------------
# LLM SELECT (on small set)
# ----------------------------
def select_node(query, nodes):
    options = "\n".join([
        f"{i}. {n.title} - {n.summary}"
        for i, n in enumerate(nodes)
    ])

    prompt = f"""
Query: {query}

Pick best section.

Options:
{options}
Return index only.
"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return int(res.choices[0].message.content.strip())
    except:
        return 0


# ----------------------------
# HYBRID SEARCH
# ----------------------------
def hybrid_search(query, bm25, mapping):
    # Step 1: Fast shortlist
    candidates = bm25_search(query, bm25, mapping, top_k=5)

    # Step 2: LLM reasoning on shortlist
    idx = select_node(query, candidates)
    return candidates[idx], candidates


# ----------------------------
# ANSWER
# ----------------------------
def generate_answer(query, node):
    prompt = f"""
Answer using document.

Query: {query}

Context:
{node.text[:4000]}
"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return res.choices[0].message.content


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    docs = load_corpus()

    all_nodes = []
    for d in docs:
        all_nodes.extend(build_nodes(d))

    print("Building BM25 index...")
    bm25, mapping = build_bm25(all_nodes)

    print("Ready!")

    while True:
        q = input("\nAsk (or exit): ")

        if q.lower() == "exit":
            break

        node, candidates = hybrid_search(q, bm25, mapping)
        answer = generate_answer(q, node)

        print("\nTop Candidates:")
        for c in candidates:
            print("-", c.title)

        print("\nAnswer:\n", answer)
