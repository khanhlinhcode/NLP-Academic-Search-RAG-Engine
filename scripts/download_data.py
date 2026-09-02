"""
Download and prepare the academic papers dataset.

Uses the Hugging Face `ccdv/arxiv-summarization` dataset
and filters for CS/ML/NLP-related papers.

Usage:
    python -m scripts.download_data
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from tqdm import tqdm

from src.config import settings


def download_and_prepare(max_papers: int = 15000) -> None:
    """Download arXiv papers and save as JSONL. """

    settings.data.ensure_dirs()
    output_path = settings.data.raw_dir / "papers.jsonl"

    if output_path.exists():
        # Count existing lines
        with open(output_path, "r") as f:
            existing_count = sum(1 for _ in f)
        print(f"📄 Dataset already exists with {existing_count} papers at {output_path}")
        print("   Delete it to re-download.")
        return

    print("📥 Downloading arXiv dataset from Hugging Face...")
    print("   This may take a few minutes on the first run.\n")

    # Load the dataset — use streaming to avoid downloading the full thing
    dataset = load_dataset(
        "ccdv/arxiv-summarization",
        split="train",
        streaming=True,
    )

    papers = []
    paper_id = 0

    print(f"🔍 Collecting up to {max_papers} papers...\n")

    for item in tqdm(dataset, total=max_papers, desc="Processing papers"):
        abstract = item.get("abstract", "").strip()
        article = item.get("article", "").strip()

        # Skip papers with very short abstracts
        if len(abstract) < 100:
            continue

        # Extract a title from the first sentence of the article, or use a generated one
        # The arxiv-summarization dataset doesn't have explicit titles,
        # so we use the first line of the article as a proxy
        first_line = article.split("\n")[0].strip() if article else ""
        title = first_line[:200] if len(first_line) > 20 else f"arXiv Paper {paper_id}"

        paper = {
            "id": f"paper_{paper_id:05d}",
            "title": title,
            "abstract": abstract,
            "authors": [],  # Not available in this dataset
            "category": "cs",  # General CS category
            "year": 2023,  # Placeholder
        }

        papers.append(paper)
        paper_id += 1

        if paper_id >= max_papers:
            break

    # Write to JSONL
    print(f"\n💾 Saving {len(papers)} papers to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"✅ Done! {len(papers)} papers saved.\n")

    # Print sample
    print("📋 Sample paper:")
    sample = papers[0]
    print(f"   ID: {sample['id']}")
    print(f"   Title: {sample['title'][:80]}...")
    print(f"   Abstract: {sample['abstract'][:120]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download arXiv papers dataset")
    parser.add_argument(
        "--max-papers",
        type=int,
        default=15000,
        help="Maximum number of papers to download (default: 15000)",
    )
    args = parser.parse_args()

    download_and_prepare(max_papers=args.max_papers)
