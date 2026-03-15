"""Generate the local 'RAG concepts in this codebase' document."""

from __future__ import annotations

import sys
from pathlib import Path

from rag_knb.concepts_documentation import (
    DEFAULT_CONCEPTS_DOC_PATH,
    write_concepts_document,
)


def main(argv: list[str] | None = None) -> int:
    """Write the concepts document to the requested path."""
    args = list(sys.argv[1:] if argv is None else argv)
    output_path = Path(args[0]) if args else DEFAULT_CONCEPTS_DOC_PATH
    written_path = write_concepts_document(output_path)
    print(written_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
