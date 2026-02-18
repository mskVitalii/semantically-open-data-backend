import hashlib
import json
from dataclasses import dataclass, asdict
from typing import ClassVar, Any

import numpy as np

from src.datasets.datasets_metadata import FieldNumeric, FieldDate, FieldString
from src.datasets_api.datasets_dto import DatasetResponse


@dataclass
class LLMQuestion:
    question: str
    reason: str

    json_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["question", "reason"],
        "additionalProperties": False,
    }

    @property
    def question_hash(self) -> str:
        return hashlib.sha256(self.question.encode()).hexdigest()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["question_hash"] = self.question_hash
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class LLMQuestionWithEmbeddings(LLMQuestion):
    embeddings: Any  # np.ndarray | SparseVector | HybridEmbedding

    def _serialize_embeddings(self):
        from src.vector_search.embedder import SparseVector, HybridEmbedding

        if isinstance(self.embeddings, np.ndarray):
            return self.embeddings.tolist()
        elif isinstance(self.embeddings, SparseVector):
            return {"indices": self.embeddings.indices, "values": self.embeddings.values}
        elif isinstance(self.embeddings, HybridEmbedding):
            return {
                "dense": self.embeddings.dense.tolist(),
                "sparse": {
                    "indices": self.embeddings.sparse.indices,
                    "values": self.embeddings.sparse.values,
                },
            }
        return self.embeddings

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "reason": self.reason,
            "question_hash": self.question_hash,
            "embeddings": self._serialize_embeddings(),
        }


@dataclass
class LLMQuestionWithDatasets(LLMQuestion):
    datasets: list[DatasetResponse]

    def to_json(self) -> str:
        data = asdict(self)
        data["question_hash"] = self.question_hash
        data["datasets"] = [ds.to_dict() for ds in self.datasets]
        data.pop("reason", None)
        return json.dumps(data)

    def to_llm_context(self) -> str:
        """Generate comprehensive context for LLM from LLMQuestionWithDatasets instance"""

        context_parts: list[str] = [
            f"\n## Research Question\n{self.question}",
            f"\n## Motivation for the question\n{self.reason}",
            f"\n## Available Datasets ({len(self.datasets)} datasets)",
        ]

        for i, dataset_response in enumerate(self.datasets, 1):
            ds = dataset_response.metadata
            context_parts.append(f"\n### Dataset {i}: {ds.title}")
            context_parts.append(f"**Relevance Score**: {dataset_response.score:.2f}")

            # Basic metadata
            if ds.description:
                context_parts.append(f"**Description**: {ds.description}")
            if ds.organization:
                context_parts.append(f"**Organization**: {ds.organization}")
            if ds.url:
                context_parts.append(f"**Source URL**: {ds.url}")
            if ds.author:
                context_parts.append(f"**Author**: {ds.author}")

            # Location information
            location_parts = []
            if ds.city:
                location_parts.append(ds.city)
            if ds.state:
                location_parts.append(ds.state)
            if ds.country:
                location_parts.append(ds.country)
            if location_parts:
                context_parts.append(f"**Location**: {', '.join(location_parts)}")

            # Temporal information
            if ds.metadata_created:
                context_parts.append(f"**Created**: {ds.metadata_created}")
            if ds.metadata_modified:
                context_parts.append(f"**Last Modified**: {ds.metadata_modified}")

            # Tags and groups
            if ds.tags:
                context_parts.append(f"**Tags**: {', '.join(ds.tags)}")
            if ds.groups:
                context_parts.append(f"**Groups**: {', '.join(ds.groups)}")

            # Fields information - CRITICAL DATA
            if ds.fields:
                context_parts.append(
                    f"\n**Dataset Structure & Fields** ({len(ds.fields)} fields):"
                )
                context_parts.append(
                    "Each field below represents a specific data dimension with its characteristics and semantic meaning:"
                )

                for field_name, field_info in ds.fields.items():
                    context_parts.append(
                        f"\n**`{field_name}`** — {field_info.type} field"
                    )

                    # Data quality indicators
                    total_values = field_info.unique_count + field_info.null_count
                    data_completeness = 100 * (
                        1 - field_info.null_count / max(total_values, 1)
                    )
                    context_parts.append(
                        f"• Completeness: {data_completeness:.1f}% complete ({field_info.null_count} nulls out of {total_values})"
                    )
                    context_parts.append(
                        f"• Cardinality: {field_info.unique_count} distinct values"
                    )

                    # Numeric field statistics with semantic interpretation
                    if isinstance(field_info, FieldNumeric):
                        context_parts.append("• **Quantitative Characteristics:**")
                        context_parts.append(
                            f"  - Central Value: Mean={field_info.mean:.2f}, Median={field_info.quantile_50_median:.2f}"
                        )
                        context_parts.append(
                            f"  - Variability: Std={field_info.std:.2f}"
                        )
                        context_parts.append(
                            f"  - Value Range: [{field_info.quantile_0_min:.2f}, {field_info.quantile_100_max:.2f}]"
                        )
                        context_parts.append(
                            f"  - Distribution Quartiles: Q1={field_info.quantile_25:.2f}, Q3={field_info.quantile_75:.2f}"
                        )

                        # Semantic interpretation based on statistics
                        if field_info.std > 0 and field_info.mean != 0:
                            cv = (field_info.std / abs(field_info.mean)) * 100
                            if cv < 15:
                                context_parts.append(
                                    "  ⇒ **Interpretation**: Values are highly consistent (low variation)"
                                )
                            elif cv < 40:
                                context_parts.append(
                                    "  ⇒ **Interpretation**: Moderate spread in values (medium variation)"
                                )
                            else:
                                context_parts.append(
                                    "  ⇒ **Interpretation**: Wide range of values (high variation)"
                                )

                        # Check for skewness
                        if (
                            abs(field_info.mean - field_info.quantile_50_median)
                            > field_info.std * 0.5
                        ):
                            context_parts.append(
                                "  ⇒ **Note**: Mean and median differ significantly - data may be skewed"
                            )

                    # Date field statistics with semantic interpretation
                    elif isinstance(field_info, FieldDate):
                        context_parts.append("• **Temporal Characteristics:**")
                        context_parts.append(
                            f"  - Time Period: {field_info.min.isoformat()} → {field_info.max.isoformat()}"
                        )
                        context_parts.append(
                            f"  - Temporal Midpoint: {field_info.mean.isoformat()}"
                        )

                        # Calculate and interpret time range
                        time_span_days = (field_info.max - field_info.min).days
                        if time_span_days < 31:
                            context_parts.append(
                                f"  ⇒ **Interpretation**: Short-term temporal data ({time_span_days} days)"
                            )
                        elif time_span_days < 365:
                            months = time_span_days // 30
                            context_parts.append(
                                f"  ⇒ **Interpretation**: Medium-term temporal data (~{months} months)"
                            )
                        else:
                            years = time_span_days / 365.25
                            context_parts.append(
                                f"  ⇒ **Interpretation**: Long-term temporal data (~{years:.1f} years)"
                            )

                    # String field with semantic interpretation
                    elif isinstance(field_info, FieldString):
                        context_parts.append("• **Categorical Characteristics:**")
                        if field_info.unique_count == 1:
                            context_parts.append(
                                "  ⇒ **Interpretation**: Constant field (single value across all records)"
                            )
                        elif field_info.unique_count < 5:
                            context_parts.append(
                                f"  ⇒ **Interpretation**: Very low cardinality ({field_info.unique_count} categories) - likely a classification/status field"
                            )
                        elif field_info.unique_count < 20:
                            context_parts.append(
                                f"  ⇒ **Interpretation**: Low cardinality ({field_info.unique_count} categories) - good for grouping/categorization"
                            )
                        elif field_info.unique_count < 100:
                            context_parts.append(
                                f"  ⇒ **Interpretation**: Medium cardinality ({field_info.unique_count} categories) - categorical with some variety"
                            )
                        else:
                            ratio = field_info.unique_count / max(total_values, 1)
                            if ratio > 0.9:
                                context_parts.append(
                                    f"  ⇒ **Interpretation**: Very high cardinality ({field_info.unique_count} values) - likely unique identifiers or free text"
                                )
                            else:
                                context_parts.append(
                                    f"  ⇒ **Interpretation**: High cardinality ({field_info.unique_count} values) - diverse categorical data"
                                )

            context_parts.append("")  # Empty line between datasets

        return "\n".join(context_parts)

    def to_llm_context_with_data(self, data_by_dataset_id: dict[str, list[dict]]) -> str:
        """Generate context that includes both field statistics and actual data rows."""
        base_context = self.to_llm_context()

        data_parts: list[str] = [base_context, "\n## Actual Data Rows"]

        for dataset_response in self.datasets:
            ds_id = dataset_response.metadata.id
            rows = data_by_dataset_id.get(ds_id)
            if not rows:
                continue

            data_parts.append(f"\n### Data from: {dataset_response.metadata.title}")
            data_parts.append(f"**Rows retrieved**: {len(rows)}")

            # Show rows as compact JSON (limit to 50 for prompt size)
            display_rows = rows[:50]
            # Remove internal fields from display
            clean_rows = []
            for row in display_rows:
                clean = {
                    k: v for k, v in row.items()
                    if k not in ("_id", "created_at", "updated_at")
                }
                clean_rows.append(clean)

            data_parts.append(f"```json\n{json.dumps(clean_rows, indent=1, ensure_ascii=False, default=str)}\n```")

            if len(rows) > 50:
                data_parts.append(f"_(showing 50 of {len(rows)} rows)_")

        return "\n".join(data_parts)
