# Labeling Rubric — SporTech Retrieval Benchmark

## Relevance Grades

| Grade | Label              | Criteria                                                                            |
|-------|--------------------|-------------------------------------------------------------------------------------|
| 3     | Highly Relevant    | Article directly answers or is specifically about the query topic, player, or event |
| 2     | Relevant           | Article discusses the same player, team, or tournament mentioned in the query       |
| 1     | Marginally Relevant| Article covers the same sport; only tangentially related to the specific query      |
| 0     | Irrelevant         | Different sport or completely unrelated content                                     |

## Query Types

- **entity**: Named entity + tournament/team queries (best for lexical methods)
- **factual**: Natural language questions with specific factual answers (best for DenseQA)
- **semantic**: Concept/theme queries without specific named entities (best for semantic)
- **hybrid**: Mixed entity + context queries (best for hybrid RRF)

## Inter-Annotator Agreement
Sanity check: compute Krippendorff's alpha across 10% double-annotated queries.
Target: α > 0.7 before trusting labels.
