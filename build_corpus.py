import os
import json
import base64
import fitz  # PyMuPDF
from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
client = OpenAI(
    base_url="YOUR_INTERNAL_OPENAI_URL",
    api_key="YOUR_API_KEY"
)

MODEL = "gpt-4o-mini"


# ----------------------------
# EXTRACT TEXT + IMAGES
# ----------------------------
def extract_pdf_multimodal(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num} ---\n{text}\n"

        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Skip tiny images (icons/logos)
            if len(image_bytes) < 5000:
                continue

            b64 = base64.b64encode(image_bytes).decode()

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail. Extract tables, numbers, and insights."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}
                            }
                        ]
                    }
                ]
            )

            image_text = response.choices[0].message.content

            full_text += f"\n[Image {img_index} Page {page_num}]\n{image_text}\n"

    return full_text


# ----------------------------
# BUILD TOC (LLM)
# ----------------------------
def build_toc(text):
    prompt = f"""
Create a structured table of contents.

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


# ----------------------------
# SAVE CORPUS
# ----------------------------
def save_corpus(pdf_path):
    os.makedirs("corpus", exist_ok=True)

    print(f"Processing {pdf_path}...")

    text = extract_pdf_multimodal(pdf_path)
    toc = build_toc(text)

    data = {
        "source": os.path.basename(pdf_path),
        "text": text,
        "toc": toc
    }

    name = os.path.splitext(os.path.basename(pdf_path))[0]
    out = f"corpus/{name}.json"

    with open(out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved: {out}")


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    pdfs = [f"pdfs/{f}" for f in os.listdir("pdfs") if f.endswith(".pdf")]

    for pdf in pdfs:
        save_corpus(pdf)
