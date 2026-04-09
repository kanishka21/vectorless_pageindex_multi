import os
import json
from openai import OpenAI

client = OpenAI(
    base_url="YOUR_INTERNAL_OPENAI_URL",
    api_key="YOUR_API_KEY"
)

MODEL = "gpt-4o-mini"


# ----------------------------
# LOAD ALL CORPUS FILES
# ----------------------------
def load_corpus(folder="corpus"):
    docs = []

    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file)) as f:
                docs.append(json.load(f))

    return docs


# ----------------------------
# NODE CLASS
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
# LLM SELECT
# ----------------------------
def select_node(query, nodes):
    options = "\n".join([
        f"{i}. {n.title} - {n.summary}"
        for i, n in enumerate(nodes)
    ])

    prompt = f"""
Query: {query}

Pick best section index.

Options:
{options}
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
# TREE SEARCH
# ----------------------------
def search_tree(query, nodes):
    path = []
    current = nodes

    while True:
        idx = select_node(query, current)
        node = current[idx]

        path.append(node)

        if not node.children:
            return node, path

        current = node.children


# ----------------------------
# FINAL ANSWER
# ----------------------------
def answer(query, node, path):
    path_text = "\n".join([n.title for n in path])

    prompt = f"""
Answer using document.

Query: {query}

Path:
{path_text}

Content:
{node.text[:4000]}
"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return res.choices[0].message.content


# ----------------------------
# MAIN QUERY LOOP
# ----------------------------
if __name__ == "__main__":
    docs = load_corpus()

    all_nodes = []
    for d in docs:
        all_nodes.extend(build_nodes(d))

    while True:
        q = input("Ask: ")

        if q == "exit":
            break

        node, path = search_tree(q, all_nodes)
        ans = answer(q, node, path)

        print("\nPath:")
        for p in path:
            print("-", p.title)

        print("\nAnswer:\n", ans)
