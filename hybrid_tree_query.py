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
        text = f"{node.title} {node.summary} {node.text[:300]}"
        tokens = text.lower().split()

        corpus.append(tokens)
        mapping.append(node)

    bm25 = BM25Okapi(corpus)
    return bm25, mapping


# ----------------------------
# BM25 SEARCH
# ----------------------------
def bm25_search(query, bm25, mapping, top_k=5):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [mapping[i] for i in top_idx]


# ----------------------------
# LLM SELECT
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
# TREE TRAVERSAL (KEY PART)
# ----------------------------
def traverse_tree(query, node):
    path = [node]

    current = node.children

    while current:
        idx = select_node(query, current)
        selected = current[idx]

        path.append(selected)
        current = selected.children

    return path[-1], path


# ----------------------------
# HYBRID SEARCH
# ----------------------------
def hybrid_tree_search(query, bm25, mapping):
    # Step 1: BM25 shortlist
    candidates = bm25_search(query, bm25, mapping, top_k=5)

    # Step 2: LLM picks best root node
    idx = select_node(query, candidates)
    best_root = candidates[idx]

    # Step 3: Tree traversal
    final_node, path = traverse_tree(query, best_root)

    return final_node, path, candidates


# ----------------------------
# ANSWER
# ----------------------------
def generate_answer(query, node, path):
    path_text = " → ".join([n.title for n in path])

    prompt = f"""
Answer using document.

Query: {query}

Reasoning path:
{path_text}

Content:
{node.text[:4000]}

Give precise answer.
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

        node, path, candidates = hybrid_tree_search(q, bm25, mapping)
        answer = generate_answer(q, node, path)

        print("\nTop BM25 Candidates:")
        for c in candidates:
            print("-", c.title)

        print("\nReasoning Path:")
        print(" → ".join([p.title for p in path]))

        print("\nAnswer:\n", answer)
