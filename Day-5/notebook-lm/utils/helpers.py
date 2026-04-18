# =============================================================================
# utils/helpers.py — Utility functions for NotebookLM Replica
# =============================================================================

from datetime import datetime
from pathlib import Path
from typing import List
from config import NOTES_DIR


def list_notes() -> List[dict]:
    """
    Return a list of saved notes with filename, timestamp, and content.
    Sorted newest first.
    """
    note_files = sorted(NOTES_DIR.glob("*.md"), reverse=True)
    notes      = []
    for f in note_files:
        try:
            content = f.read_text(encoding="utf-8")
            # Extract title from first # heading or use filename
            title = f.stem.replace("_", " ").title()
            for line in content.split("\n"):
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            notes.append({
                "filename" : f.name,
                "title"    : title,
                "content"  : content,
                "path"     : str(f),
                "modified" : datetime.fromtimestamp(f.stat().st_mtime)
                             .strftime("%Y-%m-%d %H:%M"),
            })
        except Exception:
            continue
    return notes


def download_all_notes() -> str:
    """
    Concatenate all saved notes into a single markdown string for download.
    """
    notes  = list_notes()
    if not notes:
        return "# No notes saved yet."

    parts = [
        f"# NotebookLM Notes Export\n"
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Total notes: {len(notes)}\n\n---\n"
    ]
    for note in notes:
        parts.append(f"\n## {note['title']}\n*Saved: {note['modified']}*\n\n{note['content']}\n\n---\n")

    return "\n".join(parts)


def delete_note(filename: str) -> bool:
    """Delete a note file. Returns True if deleted."""
    path = NOTES_DIR / filename
    if path.exists():
        path.unlink()
        return True
    return False


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/1024**2:.1f} MB"
