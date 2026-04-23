"""Query processor: entity recognition + optional LLM query rewrite."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ProcessedQuery:
    """Output of query processing."""

    original: str
    rewritten: Optional[str] = None
    entity_ids: list[uuid.UUID] = field(default_factory=list)

    @property
    def effective_query(self) -> str:
        """Return rewritten query if available, else original."""
        return self.rewritten if self.rewritten is not None else self.original


REWRITE_SYSTEM_PROMPT = """You are a search query optimizer. Rewrite the user's search query to be more specific and find better results. Keep the language the same (Chinese stays Chinese). Return ONLY the rewritten query, nothing else."""


class QueryProcessor:
    """Process search queries: recognize entities and optionally rewrite via LLM."""

    def __init__(
        self,
        db_session: AsyncSession,
        llm_adapter: Any | None = None,
    ):
        self._session = db_session
        self._llm_adapter = llm_adapter

    async def process(
        self,
        query: str,
        space_id: uuid.UUID,
        org_id: uuid.UUID,
        rewrite_query: bool = False,
    ) -> ProcessedQuery:
        """Process a search query: entity recognition + optional LLM rewrite.

        Steps:
        1. Extract candidate terms from the query (split on punctuation/whitespace)
        2. Match candidates against entity names in the DB
        3. Optionally rewrite the query via LLM for better search

        Args:
            query: Raw search query string.
            space_id: Space scope.
            org_id: Organization scope.
            rewrite_query: Whether to call LLM for query rewriting.

        Returns:
            ProcessedQuery with recognized entity IDs and optional rewrite.
        """
        result = ProcessedQuery(original=query)

        # ── Step 1: Entity recognition ──────────────────────────
        candidates = self._extract_candidates(query)

        if candidates:
            entity_ids = await self._recognize_entities(candidates, space_id, org_id)
            result.entity_ids = entity_ids

        # ── Step 2: Optional LLM rewrite ────────────────────────
        if rewrite_query and self._llm_adapter is not None:
            rewritten = await self._rewrite_via_llm(query)
            result.rewritten = rewritten

        return result

    def _extract_candidates(self, query: str) -> list[str]:
        """Extract candidate entity name tokens from the query.

        Strategy: Split on common Chinese/English punctuation and whitespace,
        filter tokens >= 2 chars, deduplicate while preserving order.
        """
        tokens = re.split(r'[，。！？、；：""''（）\(\)\[\]\{\}\s,.\-!?;:]+', query)
        candidates = []
        seen = set()
        for token in tokens:
            token = token.strip()
            if len(token) >= 2 and token not in seen:
                candidates.append(token)
                seen.add(token)
        return candidates

    async def _recognize_entities(
        self,
        candidates: list[str],
        space_id: uuid.UUID,
        org_id: uuid.UUID,
    ) -> list[uuid.UUID]:
        """Match candidate tokens against entity names in the DB.

        Uses ILIKE for fuzzy matching — handles partial name matches.
        """
        entity_ids: list[uuid.UUID] = []

        conditions = []
        params: dict[str, Any] = {
            "space_id": str(space_id),
            "org_id": str(org_id),
        }
        for i, candidate in enumerate(candidates):
            param_name = f"c{i}"
            conditions.append(f"e.name ILIKE :{param_name}")
            params[param_name] = f"%{candidate}%"

        if not conditions:
            return entity_ids

        where_clause = " OR ".join(conditions)
        sql = text(f"""
            SELECT e.id, e.name, e.entity_type
            FROM entities e
            WHERE e.space_id = :space_id
              AND e.org_id = :org_id
              AND ({where_clause})
        """)

        db_result = await self._session.execute(sql, params)
        for row in db_result.fetchall():
            entity_ids.append(row.id)

        return entity_ids

    async def _rewrite_via_llm(self, query: str) -> str:
        """Call LLM adapter to rewrite the query for better search results."""
        response = await self._llm_adapter.complete(
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Original query: {query}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return response.strip() if isinstance(response, str) else str(response).strip()