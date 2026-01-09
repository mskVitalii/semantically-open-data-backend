# Semantic Search Testing Parameters

## 1. Search Parameters (TestConfig)

### Embedding Models

| Parameter | Values | Recommendations |
|-----------|--------|-----------------|
| `embedder_model` | `jinaai-jina-embeddings-v3` (default)<br>`baai-bge-m3`<br>`intfloat-multilingual-e5-base`<br>`sentence-transformers-labse` | Test all models to compare quality |

### Search Parameters

| Parameter | Range | Recommendations |
|-----------|-------|-----------------|
| `limit` | 1-20 | Test: 5, 10, 15, 20 |
| `city` | string | City filter |
| `state` | string | State/region filter |
| `country` | string | Country filter |
| `use_multi_query` | true/false | Compare both options |

## 2. Qdrant Collection Settings

### HNSW Parameters (affect search quality)

| Parameter | Current | Testing Recommendations | Impact |
|-----------|---------|------------------------|--------|
| `m` | 16 | 8, 16, 32, 64 | ↑ = better quality, more memory |
| `ef_construct` | 100 | 100, 200, 400 | ↑ = better index, slower indexing |
| `full_scan_threshold` | 10000 | 1000, 10000, 50000 | Can be lower for small collections |

### Quantization (memory savings)

| Parameter | Current | Recommendations | Impact |
|-----------|---------|-----------------|--------|
| `quantization_config` | null | Test scalar int8 | ~75% memory savings, slight quality loss |

**Scalar quantization example:**
```json
{
  "scalar": {
    "type": "int8",
    "quantile": 0.99,
    "always_ram": true
  }
}
```

### Search Parameters (runtime)

| Parameter | Description | Recommendations |
|-----------|-------------|-----------------|
| `ef` | HNSW search parameter | Test: 64, 128, 256, 512<br>↑ = more accurate, slower |
| `exact` | Exact search instead of ANN | Use for baseline comparison |

**Note:** Current implementation doesn't support `ef` and `exact` in `search()` method. To use them, extend `vector_db.py`.

## 3. Evaluation Metrics

### Weighted Score

**Formula:** `avg(similarity_score * relevance_rating)`

Where:
- `similarity_score` - cosine similarity from Qdrant (0-1)
- `relevance_rating` - your manual rating (0, 0.5, 1)

**Interpretation:**
- Higher = better model finds relevant datasets with high similarity
- Considers both embedding accuracy and actual relevance

### Relevance Metrics

**Formula:** `(count(rating >= 0.5) / total) * 100%`

**Interpretation:**
- Percentage of found datasets that were relevant
- Similar to Precision in IR metrics
- Higher = less "noise" in results

## 4. Priority Parameters for Testing

### High Priority (strongly affect quality)

1. **Embedding model** - most important parameter
2. **limit** - affects result completeness
3. **use_multi_query** - can significantly improve recall

### Medium Priority (matter for large collections)

4. **m** (HNSW) - affects ANN search accuracy
5. **ef_construct** (HNSW) - index quality

### Low Priority (for optimization)

6. **quantization** - memory savings with minimal quality loss
7. **full_scan_threshold** - speed optimization

## 5. Recommended Testing Sequence

### Stage 1: Basic Model Comparison

```bash
# Test all models with same parameters
python -m src.scripts.run_benchmark --all-models --limits 10
```

Goal: Identify the best embedding model.

### Stage 2: Limit Optimization

```bash
# Test different limits for best model
python -m src.scripts.run_benchmark \
  --model [best_model] \
  --limits 5,10,15,20
```

Goal: Find optimal balance between completeness and relevance.

### Stage 3: Multi-query vs Regular Search

```bash
# Without multi-query
python -m src.scripts.run_benchmark --model [best_model] --limits [optimal_limit]

# With multi-query
python -m src.scripts.run_benchmark --model [best_model] --limits [optimal_limit] --multi-query
```

Goal: Determine if multi-query is worth using (quality improvement vs speed).

### Stage 4: (Optional) HNSW Parameters

Only if collection is large (>100k documents) and there are ANN search quality issues.

Modify via Qdrant API and reindex collection.

## 6. Testing Matrix Example

| # | Model | Limit | Multi-query | Avg Weighted Score | Relevance % | Avg Time (s) |
|---|-------|-------|-------------|-------------------|-------------|--------------|
| 1 | jinaai-jina-embeddings-v3 | 10 | No | ? | ? | ? |
| 2 | baai-bge-m3 | 10 | No | ? | ? | ? |
| 3 | intfloat-multilingual-e5-base | 10 | No | ? | ? | ? |
| 4 | sentence-transformers-labse | 10 | No | ? | ? | ? |
| 5 | [best] | 5 | No | ? | ? | ? |
| 6 | [best] | 15 | No | ? | ? | ? |
| 7 | [best] | 20 | No | ? | ? | ? |
| 8 | [best] | [optimal] | Yes | ? | ? | ? |

## 7. Quick Start

```bash
# 1. Run all tests
python -m src.scripts.run_benchmark --all-models --limits 5,10,15

# 2. Rate relevance (via API or script)

# 3. Get list of reports
curl http://localhost:8000/testing/reports

# 4. Export to Excel
python -m src.scripts.run_benchmark --export report1,report2,report3

# 5. Analyze Excel file
# - "Weighted Scores" sheet - main quality metric
# - "Relevance Metrics" sheet - precision
```

## 8. Metric Calculation Formulas

### Weighted Score (for single question)

```python
weighted_scores = [score * rating for score, rating in zip(scores, ratings)]
avg_weighted_score = sum(weighted_scores) / len(weighted_scores)
```

### Relevance Metrics (for single question)

```python
relevant_count = sum(1 for rating in ratings if rating >= 0.5)
relevance_pct = (relevant_count / len(ratings)) * 100
```

### Aggregation Across All Questions

In Excel table, each cell = metric for one question.
You can add "Average" row for mean values across all questions.

## 9. Collection Configuration via API

### Current Collection Settings

```json
{
  "status": "green",
  "optimizer_status": "ok",
  "indexed_vectors_count": 0,
  "points_count": 702,
  "segments_count": 8,
  "config": {
    "params": {
      "vectors": {
        "size": 1024,
        "distance": "Cosine"
      },
      "shard_number": 1,
      "replication_factor": 1,
      "write_consistency_factor": 1,
      "on_disk_payload": true
    },
    "hnsw_config": {
      "m": 16,
      "ef_construct": 100,
      "full_scan_threshold": 10000,
      "max_indexing_threads": 0,
      "on_disk": false
    },
    "optimizer_config": {
      "deleted_threshold": 0.2,
      "vacuum_min_vector_number": 1000,
      "default_segment_number": 0,
      "max_segment_size": null,
      "memmap_threshold": null,
      "indexing_threshold": 20000,
      "flush_interval_sec": 5,
      "max_optimization_threads": null
    },
    "quantization_config": null
  }
}
```

### Modifying HNSW Parameters

```bash
# IMPORTANT: Requires reindexing!
curl -X PATCH "http://localhost:6333/collections/{collection_name}" \
  -H "Content-Type: application/json" \
  -d '{
    "hnsw_config": {
      "m": 32,
      "ef_construct": 200
    }
  }'
```

### Adding Quantization

```bash
curl -X PATCH "http://localhost:6333/collections/{collection_name}" \
  -H "Content-Type: application/json" \
  -d '{
    "quantization_config": {
      "scalar": {
        "type": "int8",
        "quantile": 0.99,
        "always_ram": true
      }
    }
  }'
```

## 10. Testing Checklist

- [ ] Prepare test questions (test_data/questions.json)
- [ ] Run tests for all models
- [ ] Rate relevance of found datasets (0, 0.5, 1)
- [ ] Export results to Excel
- [ ] Analyze Weighted Scores
- [ ] Analyze Relevance Metrics
- [ ] Select best configuration
- [ ] (Optional) Test HNSW parameters
- [ ] (Optional) Test quantization
- [ ] Document findings
