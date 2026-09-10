"""
Deterministic PII redaction gateway.
Intercepts raw document text and sanitizes personal identifiers
before any content is sent to cloud inference APIs.

Two layers:
1. Presidio (NLP-based) -> catches contact and identifier-style PII
2. Regex fallback       -> catches patterns Presidio might miss

Researcher names, institutional names, and locations are preserved.
They are structural data for this academic profiling pipeline.
"""

import re
from dotenv import load_dotenv

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

load_dotenv()

# NLP Engine Setup

def build_analyzer() -> AnalyzerEngine:
    """
    Build Presidio analyzer using spaCy en_core_web_lg.
    This is the NLP engine that detects PII entities in text.
    """
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"}
        ]
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    return analyzer


ANALYZER  = build_analyzer()
ANONYMIZER = AnonymizerEngine()

# PII Entity Types to Redact

REDACT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "URL",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "US_SSN",
    "IBAN_CODE",
]

# Regex Patterns

REGEX_PATTERNS = {
    "EMAIL"  : r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "PHONE": r"\+?[\d\s\-\(\)]{7,15}\d",
    "URL"    : r"https?://[^\s]+",
}

# Core Redaction Logic

def presidio_redact(text: str) -> tuple[str, list[dict]]:
    """
    Run Presidio NLP-based PII detection and anonymization.
    Returns redacted text and a log of what was found.
    """
    results = ANALYZER.analyze(
        text     = text,
        entities = REDACT_ENTITIES,
        language = "en"
    )

    if not results:
        return text, []

    anonymized = ANONYMIZER.anonymize(
        text              = text,
        analyzer_results  = results,
        operators         = {
            entity: OperatorConfig("replace", {"new_value": f"[{entity}]"})
            for entity in REDACT_ENTITIES
        }
    )

    redaction_log = [
        {
            "entity_type": r.entity_type,
            "start"      : r.start,
            "end"        : r.end,
            "score"      : round(r.score, 3)
        }
        for r in results
    ]

    return anonymized.text, redaction_log


def regex_redact(text: str) -> tuple[str, list[str]]:
    """
    Regex fallback to catch patterns Presidio might miss.
    """
    redacted = text
    found    = []

    for label, pattern in REGEX_PATTERNS.items():
        matches = re.findall(pattern, redacted)
        if matches:
            found.append(f"{label}: {len(matches)} matches")
            redacted = re.sub(pattern, f"[{label}]", redacted)

    return redacted, found


def redact(text: str, doc_id: str = "unknown") -> dict:
    """
    Main entry point for PII redaction.
    Runs both Presidio and regex layers.
    Returns sanitized text and a full audit log.

    This is what every document passes through before
    being sent to any cloud inference API.
    """
    original_len = len(text)

    redacted_text, presidio_log = presidio_redact(text)
    redacted_text, regex_log    = regex_redact(redacted_text)

    final_len    = len(redacted_text)
    items_redacted = len(presidio_log) + len(regex_log)

    audit = {
        "doc_id"         : doc_id,
        "original_chars" : original_len,
        "redacted_chars" : final_len,
        "items_redacted" : items_redacted,
        "presidio_hits"  : presidio_log,
        "regex_hits"     : regex_log,
    }

    return {
        "sanitized_text": redacted_text,
        "audit"         : audit
    }


# Batch Processing

def redact_batch(docs: list[dict]) -> list[dict]:
    """
    Run redaction across a list of document dicts
    from the ingester. Adds sanitized_text and audit
    to each doc dict in place.
    """
    print(f"Running PII redaction on {len(docs)} documents\n")

    for doc in docs:
        result = redact(doc["text"], doc_id=doc["doc_id"])
        doc["sanitized_text"] = result["sanitized_text"]
        doc["pii_audit"]      = result["audit"]

        count = result["audit"]["items_redacted"]
        print(f"  {doc['doc_id'][:50]} -> {count} items redacted")

    print(f"\nRedaction complete. {len(docs)} documents sanitized.")
    return docs


# Test

if __name__ == "__main__":
    test_text = """
    Dr. John Smith from Stanford University can be reached at john.smith@stanford.edu
    or by phone at 415-555-0192. His collaborator Sarah Chen at Google Research
    published a paper on reinforcement learning. Visit https://johnsmith.com for more.
    The work was also supported by researchers at IIIT Hyderabad and Srinidhi Institute of Science and Technology.
    """

    print("Original text:")
    print(test_text)
    print("\nRunning redaction...\n")

    result = redact(test_text, doc_id="test_doc")

    print("Sanitized text:")
    print(result["sanitized_text"])
    print("\nAudit log:")
    import json
    print(json.dumps(result["audit"], indent=2))
