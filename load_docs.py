import os
from pathlib import Path

from backend.rag import load_knowledge_from_folder, DATA_FOLDER


def main():
    folder = Path(os.environ.get("ROHIT_KNOWLEDGE_FOLDER", "knowledge"))
    if not folder.exists():
        raise FileNotFoundError(f"Knowledge folder not found: {folder}")

    count = load_knowledge_from_folder(folder)
    print(f"Indexed {count} documents into vector store from {folder}")


if __name__ == "__main__":
    main()
