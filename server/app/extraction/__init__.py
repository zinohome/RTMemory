"""RTMemory Extraction Pipeline — three-layer extraction + document processing.

Layer 1: FactDetector — regex + rule-based filtering (zero cost)
Layer 2: Extractor — LLM structured extraction
Layer 3: DeepScanner — batch deep scan at conversation end
Document: DocumentProcessor — text/PDF/webpage → chunk → embed → graph-extract → index
"""

__all__ = [
    "FactDetector",
    "Extractor",
    "DeepScanner",
    "DocumentProcessor",
    "Chunker",
]


def __getattr__(name):
    if name == "FactDetector":
        from app.extraction.fact_detector import FactDetector
        return FactDetector
    elif name == "Extractor":
        from app.extraction.extractor import Extractor
        return Extractor
    elif name == "DeepScanner":
        from app.extraction.deep_scanner import DeepScanner
        return DeepScanner
    elif name == "DocumentProcessor":
        from app.extraction.document_processor import DocumentProcessor
        return DocumentProcessor
    elif name == "Chunker":
        from app.extraction.document_processor import Chunker
        return Chunker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")