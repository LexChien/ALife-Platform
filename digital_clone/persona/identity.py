"""Stable persona identity, independent of display-name slugging."""
import hashlib


def resolve_persona_id(persona: dict) -> str:
    """Explicit IDs survive display-name changes; legacy configs use the full name."""
    if "id" in persona:
        identity = persona["id"]
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("persona.id must be a nonempty string")
        return identity
    name = persona.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("persona.name must be a nonempty string")
    return "name:" + name


def persona_collection_name(persona_id: str) -> str:
    """Use an ASCII collection key without lossy normalization or truncation."""
    digest = hashlib.sha256(persona_id.encode("utf-8")).hexdigest()[:40]
    return f"clone_{digest}"
