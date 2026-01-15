# src/datasets_api/service.py
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
        self, system_prompt: str, messages: list[str] | None = None
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
            # "max_completion_tokens": max_completion_tokens,
        }

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

    async def get_research_questions(self, initial_question: str) -> list[LLMQuestion]:
        prompt = f"""Given this user question: "{initial_question}"

Generate 2-4 focused research questions that would help find relevant datasets to answer this question.

CRITICAL: Your response must be ONLY a JSON array, with absolutely no other text, formatting, or markdown.
Do NOT wrap the JSON in code blocks (```json), do NOT add explanations before or after.
Return ONLY the raw JSON array starting with [ and ending with ].

Each research question should:
- Be specific and searchable against dataset metadata
- Focus on the key data needed to answer the user's question
- Include a brief reason explaining why this data is needed
- Answer on the language of the question (or default english)

Required JSON format:
[
  {{"question": "What datasets contain X data?", "reason": "To understand Y aspect"}},
  {{"question": "Which datasets track Z metrics?", "reason": "To analyze W patterns"}}
]

Remember: Output ONLY the JSON array. No markdown, no code blocks, no explanations."""
        result = await self.openai_by_api(
            system_prompt=self.system_prompt, messages=[prompt]
        )
        try:
            valid_result = extract_json(result)
            return [LLMQuestion(**item) for item in valid_result]
        except Exception as e:
            logger.error(f"Failed to parse research questions: {e}", exc_info=True)
            raise

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
