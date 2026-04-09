import os
import json
from pypdf import PdfReader
from openai import OpenAI

client = OpenAI(
    base_url="YOUR_INTERNAL_OPENAI_URL",
    api_key="YOUR_API_KEY"
)

MODEL = "gpt-4o-mini"


def load_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    return text


def build_toc(text):
    prompt = f"""
Extract a structured table of contents.

Return JSON:
[
  {{
    "title": "...",
    "summary": "...",
    "subsections": [
      {{"title": "...", "summary": "..."}}
    ]
  }}
]

Document:
{text[:15000]}
"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = res.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except:
        return json.loads(content.replace("```json", "").replace("```", ""))


def save_corpus(pdf_path, output_dir="corpus"):
    os.makedirs(output_dir, exist_ok=True)

    text = load_pdf(pdf_path)
    toc = build_toc(text)

    data = {
        "source": os.path.basename(pdf_path),
        "text": text,
        "toc": toc
    }

    name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_path = f"{output_dir}/{name}.json"

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    pdfs = ["pdfs/file1.pdf", "pdfs/file2.pdf"]

    for pdf in pdfs:
        save_corpus(pdf)
