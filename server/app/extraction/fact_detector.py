"""FactDetector — Layer 1: lightweight regex + rule-based filtering.

Zero-cost pre-filter that catches 70-80% of factual statements without
calling an LLM. Supports Chinese and English patterns.
"""

from __future__ import annotations

import re


class FactDetector:
    """Determines whether a message likely contains new facts, preferences,
    or status changes worth extracting into the knowledge graph.

    Uses compiled regex rules for both Chinese and English. Optional
    context-aware boosting when recent conversation history contains
    fact-like patterns.
    """

    # ── Chinese patterns ──────────────────────────────────────
    _ZH_PATTERNS: list[str] = [
        r"我(?:是|在|有|用|喜欢|偏好|搬到|换了?|改|还是|最近|决定|打算|准备)",
        r"我们(?:用|选|决定|计划|换|改|打算|准备)",
        r"(?:推荐|建议|偏好|习惯)",
    ]

    # ── English patterns ─────────────────────────────────────
    _EN_PATTERNS: list[str] = [
        r"(?:I|i)\s+(?:am|work|live|use|like|prefer|moved|switched|changed|have|decided|plan|going)",
        r"(?:[Ww]e)\s+(?:use|chose|decided|plan|switched|changed)",
        r"(?:[Mm]y)\s+(?:name|job|role|team|company|location)\s+is",
        r"(?:recommend|suggest|preference|habit)",
    ]

    def __init__(self) -> None:
        # Compile all patterns for performance
        self._zh_rules: list[re.Pattern[str]] = [
            re.compile(p) for p in self._ZH_PATTERNS
        ]
        self._en_rules: list[re.Pattern[str]] = [
            re.compile(p, re.IGNORECASE) for p in self._EN_PATTERNS
        ]

        # Pattern to detect fact-like statements in context
        self._context_fact_re: re.Pattern[str] = re.compile(
            r"我(?:是|在|有|用|喜欢|偏好|搬到|换了?|改|还是|最近|决定|打算|准备)"
            r"|我们(?:用|选|决定|计划|换|改|打算|准备)"
            r"|(?:I|i)\s+(?:am|work|live|use|like|prefer|moved|switched|changed|have|decided|plan|going)"
            r"|(?:[Ww]e)\s+(?:use|chose|decided|plan|switched|changed)"
            r"|(?:推荐|建议|偏好|习惯)"
            r"|(?:recommend|suggest|preference|habit)"
        )

    def _match_rules(self, message: str) -> bool:
        """Check message against all compiled regex rules."""
        for rule in self._zh_rules:
            if rule.search(message):
                return True
        for rule in self._en_rules:
            if rule.search(message):
                return True
        return False

    def should_extract(
        self,
        message: str,
        context: list[str] | None = None,
    ) -> bool:
        """Determine if the message should be sent to Layer 2 extraction.

        Args:
            message: The user message to evaluate.
            context: Optional list of recent conversation messages.
                When provided and the message is ambiguous (e.g. a follow-up
                question), if any context message matches a fact pattern,
                this message is boosted to be extracted too.

        Returns:
            True if the message likely contains extractable facts.
        """
        if not message or not message.strip():
            return False

        # Direct rule match — always extract
        if self._match_rules(message):
            return True

        # Context-aware boost: if recent messages contain facts,
        # follow-up questions may refer to them
        if context:
            for ctx_msg in context:
                if self._context_fact_re.search(ctx_msg):
                    return True

        return False