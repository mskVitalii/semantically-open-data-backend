# src/datasets_api/service.py
import json
from typing import Optional

import aiohttp
from aiohttp import TCPConnector, ClientTimeout

from src.domain.services.llm_dto import LLMQuestion, LLMQuestionWithDatasets
from src.infrastructure.config import LLM_URL, LLM_OPEN_AI_KEY
from src.infrastructure.logger import get_prefixed_logger
from src.utils.llm_utils import extract_json

logger = get_prefixed_logger(__name__, "LLM_SERVICE")


class LLMService:
    """Service for working with LLM"""

    # region INIT

    def __init__(self, base_url: str):
        self.base_url = base_url
        # Create connector with connection pooling
        connector = TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,  # DNS cache for 5 minutes
            enable_cleanup_closed=True,
            force_close=True,
        )

        # Optimized timeout settings
        timeout = ClientTimeout(
            total=1200,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=1190,  # Socket read timeout
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "LLM Asker (Python/aiohttp)",
                "Accept-Encoding": "gzip, deflate",  # Enable compression
            },
        )

    async def close_llm_session(self):
        logger.info("LLMService >> __aexit__")
        if self.session:
            await self.session.close()

    # endregion

    # region LOGIC

    async def ollama_by_api(self, prompt: str, temperature=0.5, max_tokens=10):
        url = self.base_url + "/api/generate"

        data = {
            "model": "gemma3:4b",
            "prompt": prompt,
            "options": {"temperature": temperature, "max_tokens": max_tokens},
            "stream": False,
        }

        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                response_json = await response.json()
                return extract_json(response_json["response"])
            else:
                error = await response.text()
                raise Exception(f"Error {response.status}: {error}")

    async def openai_by_api(
        self,
        system_prompt: str,
        messages: list[str] | None = None,
        response_format: dict | None = None,
    ):
        if messages is None:
            messages = []

        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {LLM_OPEN_AI_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-5-nano",
            "messages": (
                [{"role": "system", "content": system_prompt}]
                + [{"role": "user", "content": m} for m in messages]
            ),
        }

        if response_format is not None:
            data["response_format"] = response_format

        async with self.session.post(url, json=data, headers=headers) as response:
            if response.status == 200:
                response_json = await response.json()
                return response_json["choices"][0]["message"]["content"]
            else:
                error = await response.text()
                raise Exception(f"Error {response.status}: {error}")

    # endregion

    # region REQUESTS

    system_prompt = """You are an urban data researcher specializing in city analytics and evidence-based urban planning. Your task is to provide data-driven insights based on available datasets from Chemnitz, Saxony, Germany. Always ground your analysis in the specific data provided and clearly reference which datasets support your conclusions."""

    _research_questions_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "research_questions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": LLMQuestion.json_schema,
                    }
                },
                "required": ["questions"],
                "additionalProperties": False,
            },
        },
    }

    async def get_research_questions(self, initial_question: str) -> list[LLMQuestion]:
        prompt = f"""Given this user question: "{initial_question}"

Generate 2-4 focused research questions that would help find relevant datasets to answer this question.

Each research question should:
- Be specific and searchable against dataset metadata
- Focus on the key data needed to answer the user's question
- Include a brief reason explaining why this data is needed
- Answer on the language of the question (or default english)"""

        result = await self.openai_by_api(
            system_prompt=self.system_prompt,
            messages=[prompt],
            response_format=self._research_questions_format,
        )
        parsed = json.loads(result)
        return [LLMQuestion(**item) for item in parsed["questions"]]

    async def answer_research_question(
        self, question: LLMQuestionWithDatasets
    ) -> list[str]:
        context = question.to_llm_context()
        instructions = """
## How to Analyze the Datasets

The datasets above include detailed field information with semantic interpretations. Use this information to provide meaningful insights:

**Understanding Field Interpretations:**
- Each field has a "⇒ Interpretation" section that explains what the data characteristics mean
- For numeric fields: Look at variation levels (low/medium/high) and skewness notes
- For temporal fields: Consider the time span (short/medium/long-term)
- For categorical fields: Use cardinality interpretations to understand data granularity

**Analysis Approach:**
1. Focus on fields most relevant to answering the research question
2. Cite specific field names and their semantic interpretations from the dataset structure
3. Use the statistical measures (mean, median, range) to quantify your findings
4. Consider data completeness percentages when assessing reliability
5. Note any data quality issues (high null counts, skewness) that affect conclusions

**Response Format:**
- Write a single focused paragraph (3-5 sentences maximum)
- Start by directly addressing the research question
- Reference specific dataset names and field names
- Use quantitative evidence from the field statistics
- If datasets lack relevant fields or have poor data quality, state "No suitable datasets found" and explain why

**Critical Rules:**
- ONLY use information explicitly present in the dataset fields and their interpretations
- DO NOT make assumptions beyond what the data shows
- DO NOT suggest what "could be" analyzed - only what IS in the data
- If the answer cannot be determined from available fields, clearly state this
- ALWAYS answer on the language of the question (or default english)
            """
        prompt = f"""{context}\n{instructions}"""

        try:
            result = await self.openai_by_api(
                system_prompt=self.system_prompt, messages=[prompt]
            )
            return result
        except Exception as e:
            logger.error(f"Failed to answer the research question: {e}", exc_info=True)
            raise

    # region MONGO FILTERS

    _mongo_filters_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "mongo_filters",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "field": {"type": "string"},
                                "operator": {
                                    "type": "string",
                                    "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "in", "regex"],
                                },
                                "value": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "number"},
                                        {"type": "boolean"},
                                        {"type": "array", "items": {"type": "string"}},
                                        {"type": "array", "items": {"type": "number"}},
                                    ]
                                },
                            },
                            "required": ["field", "operator", "value"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["filters"],
                "additionalProperties": False,
            },
        },
    }

    @staticmethod
    def describe_fields_for_filter(metadata) -> str:
        """Build a compact field description for the filter-generation prompt."""
        lines: list[str] = []
        if not metadata.fields:
            return "No field information available."
        for name, info in metadata.fields.items():
            parts = [f"- {name} ({info.type})"]
            parts.append(f"unique={info.unique_count}, nulls={info.null_count}")
            from src.datasets.datasets_metadata import FieldNumeric, FieldDate, FieldString

            if isinstance(info, FieldNumeric):
                parts.append(
                    f"range=[{info.quantile_0_min}, {info.quantile_100_max}], mean={info.mean:.2f}"
                )
            elif isinstance(info, FieldDate):
                parts.append(f"range=[{info.min.isoformat()}, {info.max.isoformat()}]")
            lines.append(", ".join(parts))
        return "\n".join(lines)

    async def get_mongo_filters(
        self,
        question: str,
        dataset_title: str,
        fields_info: str,
    ) -> list[dict]:
        """Ask ChatGPT to generate MongoDB filters for a research question."""
        prompt = f"""Given this research question: "{question}"
And this dataset: "{dataset_title}"
With these fields:
{fields_info}

Generate MongoDB-style filters to retrieve the most relevant rows for answering the research question.
Rules:
- Only use fields that exist in the dataset
- Use operator "eq" for exact match, "regex" for partial text match (case-insensitive)
- Use "in" with a list of values for multiple matches
- Return an empty filters array if no filtering is needed (the whole dataset is relevant)
- Keep filters minimal — only filter when it clearly helps answer the question"""

        result = await self.openai_by_api(
            system_prompt=self.system_prompt,
            messages=[prompt],
            response_format=self._mongo_filters_format,
        )
        parsed = json.loads(result)
        return parsed["filters"]

    @staticmethod
    def build_mongo_query(filters: list[dict]) -> dict:
        """Convert a list of {field, operator, value} dicts into a MongoDB query."""
        if not filters:
            return {}

        op_map = {
            "eq": "$eq",
            "ne": "$ne",
            "gt": "$gt",
            "gte": "$gte",
            "lt": "$lt",
            "lte": "$lte",
            "in": "$in",
        }

        query: dict = {}
        for f in filters:
            field = f["field"]
            operator = f["operator"]
            value = f["value"]

            if operator == "regex":
                query[field] = {"$regex": value, "$options": "i"}
            elif operator in op_map:
                query.setdefault(field, {})[op_map[operator]] = value
            else:
                query[field] = value

        return query

    async def answer_research_question_with_data(
        self, question: LLMQuestionWithDatasets, data_by_dataset_id: dict[str, list[dict]]
    ) -> str:
        context = question.to_llm_context_with_data(data_by_dataset_id)
        instructions = """
## How to Analyze the Datasets

You have access to both field statistics AND actual data rows from the datasets. Use the real data to give concrete, evidence-based answers.

**Analysis Approach:**
1. Examine the actual data rows to find specific values, patterns, and examples
2. Use field statistics for overall context (ranges, distributions, completeness)
3. Cite specific data points and values from the rows
4. If the data rows are a filtered subset, note that your analysis is based on a sample

**Response Format:**
- Write a single focused paragraph (3-5 sentences maximum)
- Start by directly addressing the research question
- Reference specific values and examples from the data rows
- Use quantitative evidence wherever possible
- If the data doesn't contain relevant information, clearly state this

**Critical Rules:**
- ONLY use information explicitly present in the data
- Prefer citing actual data rows over just statistics
- DO NOT make assumptions beyond what the data shows
- ALWAYS answer in the language of the question (or default English)
        """
        prompt = f"{context}\n{instructions}"

        try:
            result = await self.openai_by_api(
                system_prompt=self.system_prompt, messages=[prompt]
            )
            return result
        except Exception as e:
            logger.error(f"Failed to answer research question with data: {e}", exc_info=True)
            raise

    # endregion

    async def summary(self, messages: list[str]):
        prompt = "Summarize those paragraphs. Use 1 paragraph in answer"

        try:
            result = await self.openai_by_api(
                system_prompt=self.system_prompt, messages=messages + [prompt]
            )
            return result
        except Exception as e:
            logger.error(f"Failed to answer the research question: {e}", exc_info=True)
            raise

    # endregion


# region DI
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get MongoDB manager instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(base_url=LLM_URL)
    return _llm_service


async def get_llm_service_dep() -> LLMService:
    """Dependency to get database"""
    return get_llm_service()


# endregion
