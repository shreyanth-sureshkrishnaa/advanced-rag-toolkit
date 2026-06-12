import pathlib
import wikipediaapi

TOPICS = [
    "Vector database",
    "Retrieval-augmented generation",
    "Transformer (machine learning model)",
    "BERT (language model)",
    "Cosine similarity",
    "Information retrieval",
    "Large language model",
    "Semantic search",
    "Relational database",
    "Approximate nearest neighbor search",
    "Word embedding",
    "Sentence embedding",
]

DOCS_DIR = pathlib.Path("docs")


def fetch_all():
    wiki = wikipediaapi.Wikipedia("rag-advanced/1.0 (shreyanth@example.com)", "en")
    DOCS_DIR.mkdir(exist_ok=True)

    fetched, skipped = 0, 0
    for topic in TOPICS:
        page = wiki.page(topic)
        if not page.exists():
            print(f"  [skip] Not found: {topic}")
            skipped += 1
            continue

        slug = topic.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        out_path = DOCS_DIR / f"{slug}.md"
        out_path.write_text(f"# {topic}\n\n{page.text}", encoding="utf-8")
        print(f"  [ok]   {out_path.name}  ({len(page.text):,} chars)")
        fetched += 1

    print(f"\nDone. {fetched} articles saved, {skipped} skipped.")


if __name__ == "__main__":
    fetch_all()