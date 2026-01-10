# Semantic Search Testing Guide

This document describes the testing system for semantic search quality in Qdrant.

## Table of Contents

1. [Introduction](#introduction)
2. [Testing Parameters](#testing-parameters)
3. [Qdrant Collection Settings](#qdrant-collection-settings)
4. [Running Tests](#running-tests)
5. [Relevance Rating](#relevance-rating)
6. [Excel Export](#excel-export)
7. [Usage Examples](#usage-examples)

## Introduction

The testing system allows you to:
- Test different embedding models
- Compare search parameters
- Manually rate the relevance of found datasets
- Get Excel reports with quality metrics

## Testing Parameters

### 1. Embedding Models

Available models (from `src/infrastructure/config.py`):

| Model | Dimension | Description |
|-------|-----------|-------------|
| `baai-bge-m3` | 1024 | Multilingual BGE-M3 |
| `intfloat-multilingual-e5-base` | 768 | E5 Multilingual Base |
| `jinaai-jina-embeddings-v3` | 1024 | Jina AI v3 (default) |
| `sentence-transformers-labse` | 768 | LaBSE from Sentence Transformers |

### 2. Search Parameters

Available in `TestConfig`:

- **embedder_model**: embedding model to use (default: jinaai-jina-embeddings-v3)
- **limit** (1-25): number of results to return
- **use_multi_query** (bool): use multi-query RAG (generate multiple query variants)

### 3. Location Filters

Location filters (city, state, country) are stored **in questions** themselves, not in TestConfig.

Each test configuration is **automatically tested in 2 variants**:
1. **WITH location filters** - uses filters from question (e.g., city="Berlin")
2. **WITHOUT location filters** - searches globally without location constraints

This allows direct comparison of filtered vs. unfiltered search results.

### 4. Qdrant Collection Settings (Advanced)

To modify collection parameters, use the Qdrant API:

#### HNSW Parameters (affect search quality):

```json
{
  "hnsw_config": {
    "m": 16,              // Links between nodes: 8, 16, 32, 64
    "ef_construct": 100,  // Indexing accuracy: 100, 200, 400
    "full_scan_threshold": 10000
  }
}
```

**Recommendations:**
- **m**: higher = better quality, more memory. Test: 8, 16, 32, 64
- **ef_construct**: higher = better index, slower indexing. Test: 100, 200, 400

#### Quantization (reduces memory):

```json
{
  "quantization_config": {
    "scalar": {
      "type": "int8",
      "quantile": 0.99,
      "always_ram": true
    }
  }
}
```

## Running Tests

**Important:** Each configuration is automatically tested in **2 variants** (with/without location filters).
So if you test 16 questions with 1 config = 32 tests total (16 × 1 × 2).

### Option 1: Via CLI (Recommended)

```bash
# Test single model with one limit
# This will create 2 variants per question: with filters + without filters
python -m src.scripts.run_benchmark --model jinaai-jina-embeddings-v3 --limits 10

# Test all models
python -m src.scripts.run_benchmark --all-models --limits 10

# Test one model with different limits
python -m src.scripts.run_benchmark --model baai-bge-m3 --limits 5,10,25

# Test specific questions only
python -m src.scripts.run_benchmark --questions 0,1,2 --model jinaai-jina-embeddings-v3

# With verbose output
python -m src.scripts.run_benchmark --all-models --limits 10 --verbose
```

### Option 2: Via API

```bash
# Run tests
curl -X POST "http://localhost:8000/testing/run" \
  -H "Content-Type: application/json" \
  -d '{
    "question_indices": null,
    "test_configs": [
      {
        "embedder_model": "jinaai-jina-embeddings-v3",
        "limit": 10,
        "use_multi_query": false
      },
      {
        "embedder_model": "baai-bge-m3",
        "limit": 10,
        "use_multi_query": false
      }
    ]
  }'
```

### Option 3: Quick Test (Simplified API Endpoint)

```bash
curl -X POST "http://localhost:8000/testing/run/quick?limit=10&use_multi_query=false"
```

## Relevance Rating

After running tests, you need to manually rate the relevance of found datasets:

### Rating Scale:

- **0** - dataset is not relevant (doesn't fit)
- **0.5** - dataset is partially relevant (partially fits)
- **1** - dataset is fully relevant (fits)

### Update Rating via API:

```bash
curl -X POST "http://localhost:8000/testing/reports/{report_id}/relevance" \
  -H "Content-Type: application/json" \
  -d '{
    "report_id": "your-report-id",
    "question": "What are the safest neighborhoods...",
    "dataset_id": "dataset-123",
    "relevance_rating": 1
  }'
```

### Python Script Example for Batch Rating:

```python
import requests

# After getting the report
report_id = "your-report-id"
base_url = "http://localhost:8000"

# Rate each dataset
ratings = [
    {"question": "...", "dataset_id": "...", "relevance_rating": 1},
    {"question": "...", "dataset_id": "...", "relevance_rating": 0.5},
    # ...
]

for rating in ratings:
    response = requests.post(
        f"{base_url}/testing/reports/{report_id}/relevance",
        json={
            "report_id": report_id,
            **rating
        }
    )
    print(f"Updated: {rating['dataset_id']}")
```

## Excel Export

After rating relevance, export results to Excel for analysis:

### Via CLI:

```bash
# Export multiple reports
python -m src.scripts.run_benchmark --export report1,report2,report3

# With custom output file
python -m src.scripts.run_benchmark --export report1,report2 --output comparison.xlsx
```

### Via API:

```bash
curl "http://localhost:8000/testing/export/excel?report_ids=report1,report2,report3" \
  --output comparison.xlsx
```

### Excel File Structure:

**Filename format:** `YYYY_MM_DD_HH_MM_<report_ids>_comparison.xlsx`

Example: `2026_01_09_21_37_5a7e46_abc123_comparison.xlsx`

The file contains **3 sheets** with color-coded cells (🔴 Red → 🟡 Yellow → 🟢 Green):

**Note:** Colors are applied directly to cells for compatibility with Mac Numbers and other spreadsheet applications.

#### 1. Weighted Scores

Average weighted score: `avg(score * relevance_rating)`

| Question | jinaai-jina-embeddings-v3_limit10 | baai-bge-m3_limit10 | ... |
|----------|-----------------------------------|---------------------|-----|
| What are the safest... | 0.5197 🟡 | 0.4856 🟠 | ... |
| Where can I find... | 0.5820 🟢 | 0.5505 🟡 | ... |

**Color Scale:** Red (0.0) → Yellow (0.5) → Green (1.0)

**Interpretation:**
- Higher value = better model finds relevant datasets with high score
- Value considers both similarity score and your relevance rating
- Green cells = excellent performance, Red cells = poor performance

#### 2. Relevance Metrics

Percentage of relevant datasets (rating >= 0.5)

| Question | jinaai-jina-embeddings-v3_limit10 | baai-bge-m3_limit10 | ... |
|----------|-----------------------------------|---------------------|-----|
| What are the safest... | 80.0% 🟢 | 60.0% 🟡 | ... |
| Where can I find... | 100.0% 🟢 | 80.0% 🟢 | ... |

**Color Scale:** Red (0%) → Yellow (50%) → Green (100%)

**Interpretation:**
- Shows what percentage of found datasets was relevant
- Helps assess model precision
- Higher percentages (green) = fewer irrelevant results

#### 3. Detailed Ratings

Detailed table with all datasets and their ratings

| Question | Experiment | Dataset | Dataset ID | Score | Relevance Rating |
|----------|------------|---------|------------|-------|------------------|
| What are the safest... | jinaai-jina-embeddings-v3_limit10 | Kriminalitätsatlas Berlin | 658c7ebb... | 0.5197 🟡 | 1 |
| What are the safest... | jinaai-jina-embeddings-v3_limit10 | Straftaten (Jahreszahlen...) | 616d67a3... | 0.4564 🟠 | 0.5 |
| What are the safest... | baai-bge-m3_limit10 | Kriminalitätsatlas Berlin | 658c7ebb... | 0.4856 🟠 | 1 |
| ... | ... | ... | ... | ... | ... |

**Color Scale for Score column:** Red (0.0) → Yellow (0.5) → Green (1.0)

**Interpretation:**
- Complete list of all found datasets with their ratings
- Score column is color-coded to quickly identify high/low similarity matches
- Useful for reviewing and quality control of your relevance ratings
- Can be filtered and sorted in Excel for analysis

**File Naming:**
- Single report: `report_{report_id}.xlsx` (e.g., `report_f77852a8.xlsx`)
- Multiple reports: `comparison_{id1}_{id2}_{id3}.xlsx`

## Usage Examples

### Example 1: Compare All Models

```bash
# 1. Run tests for all models
python -m src.scripts.run_benchmark --all-models --limits 10 --verbose

# 2. System will create reports and show their IDs

# 3. Rate dataset relevance via API or script

# 4. Export to Excel (automatically created for multiple configurations)
```

### Example 2: Optimize limit Parameter

```bash
# Test different limit values for one model
python -m src.scripts.run_benchmark \
  --model jinaai-jina-embeddings-v3 \
  --limits 5,10,15,20

# After rating relevance, analyze which limit gives best results
```

### Example 3: Testing with Filters

```bash
# Test search only for Berlin
python -m src.scripts.run_benchmark \
  --model jinaai-jina-embeddings-v3 \
  --limits 10 \
  --city Berlin
```

### Example 4: Multi-query RAG

```bash
# Compare regular search and multi-query
python -m src.scripts.run_benchmark \
  --model jinaai-jina-embeddings-v3 \
  --limits 10

python -m src.scripts.run_benchmark \
  --model jinaai-jina-embeddings-v3 \
  --limits 10 \
  --multi-query

# Then export both reports for comparison
python -m src.scripts.run_benchmark --export report1,report2
```

## Complete Workflow

1. **Preparation:**
   ```bash
   # Ensure dependencies are installed
   pip install pandas openpyxl
   ```

2. **Run tests:**
   ```bash
   # Test all models
   python -m src.scripts.run_benchmark --all-models --limits 10
   ```

3. **Get report IDs:**
   ```bash
   # Reports are automatically saved to test_data/reports/
   # IDs will be shown in console
   ```

4. **View reports:**
   ```bash
   # Via API
   curl http://localhost:8000/testing/reports
   curl http://localhost:8000/testing/reports/{report_id}
   ```

5. **Rate relevance:**
   - Review found datasets for each question
   - Rate each dataset on scale 0, 0.5, 1
   - Update ratings via API

6. **Export and analyze:**
   ```bash
   # Export to Excel
   python -m src.scripts.run_benchmark --export report1,report2,report3

   # Open Excel file and analyze:
   # - Which model gives best weighted scores?
   # - Which model finds more relevant datasets?
   # - What is the optimal limit?
   ```

## Result Analysis Tips

1. **Weighted Scores**: Focus on this metric to assess overall search quality
2. **Relevance Metrics**: Use to understand model precision
3. **Compare by questions**: Some models may work better for specific question types
4. **Consider speed**: Check `execution_time_seconds` in reports

## Troubleshooting

### Issue: No pandas/openpyxl

```bash
pip install pandas openpyxl
```

### Issue: Report not found

```bash
# Check available reports
curl http://localhost:8000/testing/reports
```

### Issue: Export error

- Ensure you've rated relevance for at least some datasets
- Verify that report_ids are correct

## API Reference

Full API documentation available at: `http://localhost:8000/docs`

### Main endpoints:

- `POST /testing/questions` - Add test question (with optional location filters)
- `GET /testing/questions` - Get all questions
- `POST /testing/run` - Run bulk test
- `POST /testing/run/quick` - Quick test with single configuration
- `GET /testing/reports` - List all reports
- `GET /testing/reports/{report_id}` - Get specific report
- `POST /testing/reports/{report_id}/relevance` - Update relevance rating
- `GET /testing/export/excel` - Export to Excel

### Adding Questions with Location Filters

```bash
# Add question with city filter
curl -X POST "http://localhost:8000/v1/testing/questions" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are crime statistics in Berlin?",
    "city": "Berlin",
    "state": null,
    "country": "Germany"
  }'

# Add question with full location
curl -X POST "http://localhost:8000/v1/testing/questions" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What building permits were issued in Dresden?",
    "city": "Dresden",
    "state": "Saxony",
    "country": "Germany"
  }'

# Add question without filters
curl -X POST "http://localhost:8000/v1/testing/questions" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are open data portals in Europe?"
  }'
```
