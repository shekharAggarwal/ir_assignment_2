# Dataset card: Amazon Reviews 2023, six-department sample

| Item | Value |
|---|---|
| Source | Amazon Reviews 2023, McAuley Lab, UC San Diego: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 (`raw/review_categories/*.jsonl`) |
| Citation | Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. (2024). *Bridging Language and Items for Retrieval and Recommendation*. arXiv:2403.03952 |
| Built by | `scripts/build_dataset.py` (seed 537), built 2026-11-01 |
| Files | `amazon_reviews_corpus.json` (text and metadata stored separately), `amazon_reviews_corpus.csv` (same data, flat) |
| Documents | 1,224 reviews. Appliances 201 · Beauty 204 · Musical Instruments 203 · Pet Supplies 207 · Software 206 · Video Games 203 |
| Raw text | 0.47 MB, 88,004 whitespace tokens, 30-344 words per review |

## Sampling procedure
1. For each of the six category files, 40 byte ranges of 64 KB were read at seeded random offsets (HTTP Range
   requests), so the sample is spread over the whole file. Partial lines at the edges of each range were dropped.
2. This gave a pool of 33,856 raw reviews. 11,306 of them were kept: 30-350 words and English-looking.
3. At most 2 reviews per user were kept, because the files are ordered by user.
4. About 200 reviews per category were drawn.

## Duplicates (ground truth in `dup_group`, `dup_type`, `is_injected`)
* In the pool, 133 groups of naturally re-posted reviews were found. They cover 287 of 11,306 pooled reviews (2.5%).
  Members of a group come from the same user on the same date with (almost) the same text, usually for the same
  product listing. 36 of these groups (6 per category, size 2-3) were kept.
* 24 near-duplicate variants of other sampled reviews were added and marked `is_injected = True`. One third
  re-post with ALL-CAPS first sentence and `<br />` markup, one third add punctuation and emoticons, and one third
  are edited re-posts with two words removed and an "UPDATE" sentence appended.
* After the final IDs were assigned, ground-truth groups were recomputed over the whole corpus with union-find over:
  identical normalised text, word-3-shingle Jaccard ≥ 0.8, and the injected links.
  Result: 60 groups, 121 documents (9.9%).

## Fields
`doc_id`, `text`, `title`, `category` (label), `rating` (1-5), `date`, `helpful_votes`, `verified_purchase`,
`has_images`, `asin`, `parent_asin`, `user_id` (anonymised by the dataset authors), `source`, `dup_group`,
`dup_type` (exact = identical after normalisation, near), `is_injected`.

## Intended use
Academic coursework (BITS WILP Information Retrieval, Assignment 2). The source dataset is published by its
authors for research use.
