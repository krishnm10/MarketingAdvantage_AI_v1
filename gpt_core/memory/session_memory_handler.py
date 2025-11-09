import json
from pathlib import Path

class SessionMemory:
    """Simple short-term memory stored in local file (extendable to DB)."""
    def __init__(self, file_path: str = "session_memory.json", limit: int = 10):
        self.file = Path(file_path)
        self.limit = limit
        self.messages = []
        self._load()

    def _load(self):
        if self.file.exists():
            try:
                self.messages = json.loads(self.file.read_text(encoding="utf-8"))
            except Exception:
                self.messages = []

    def _save(self):
        self.file.write_text(json.dumps(self.messages[-self.limit:], indent=2))

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._save()

    def context(self):
        return self.messages[-self.limit:]

# Usage:
# memory = SessionMemory()
# memory.add("user", "Generate ad copy")
# print(memory.context())
