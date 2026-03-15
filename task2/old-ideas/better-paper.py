import os
import re
import ast
import jsonlines
import random
import argparse

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

argparser = argparse.ArgumentParser()
argparser.add_argument("--stage",        type=str, default="practice", help="Stage of the project")
argparser.add_argument("--lang",         type=str, default="python",   help="Language")
argparser.add_argument("--strategy",     type=str, default="random",
                       choices=["random", "bm25", "recent", "embeddings",
                                "smart_multi", "path_distance", "lines_iou", "best"],
                       help="Context collection strategy")
argparser.add_argument("--trim-prefix",  action="store_true",       help="Trim the prefix for submission")
argparser.add_argument("--trim-suffix",  action="store_true",       help="Trim the suffix for submission")
argparser.add_argument("--prefix-mode",  type=str, default="smart", choices=["smart", "simple", "full"])
argparser.add_argument("--suffix-mode",  type=str, default="full",  choices=["full", "filtered", "none"])
# shared multi-file params
argparser.add_argument("--chunk-lines",  type=int, default=30,   help="Lines per chunk for chunk-based strategies")
argparser.add_argument("--top-chunks",   type=int, default=3,    help="Top-K chunks to keep per file")
argparser.add_argument("--max-files",    type=int, default=5,    help="Max files to include in context")
argparser.add_argument("--token-budget", type=int, default=6000, help="Approximate token budget for context")

args = argparser.parse_args()

stage    = args.stage
language = args.lang
strategy = args.strategy

if language == "python":
    extension = ".py"
elif language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

print(f"Running the '{strategy}' strategy  |  stage='{stage}'  |  lang='{language}'")

FILE_SEP_SYMBOL    = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

embed_model = (
    SentenceTransformer('all-MiniLM-L6-v2')
    if strategy in ("embeddings", "smart_multi", "best")
    else None
)


# ===========================================================================
# PREFIX / SUFFIX UTILITIES
# ===========================================================================

def _module_name_from_line(line: str) -> str | None:
    """
    Extract the top-level module name from an import line.
      'import os'                → 'os'
      'from utils.auth import X' → 'utils'   (first component)
      'import torch, numpy'      → 'torch'   (first only — good enough for lookup)
    Returns None if the line isn't a recognisable import.
    """
    line = line.strip()
    if line.startswith("from "):
        parts = line[5:].split()
        return parts[0].split(".")[0] if parts else None
    if line.startswith("import "):
        parts = line[7:].split()
        return parts[0].split(".")[0].rstrip(",") if parts else None
    return None


def is_local_import(line: str, root_dir: str) -> bool:
    """
    Return True if the import line refers to a module that exists inside root_dir.
    Checks both  <module>.py  and  <module>/__init__.py  so packages work too.
    A line that can't be parsed as an import is treated as non-import (False).
    """
    mod = _module_name_from_line(line)
    if mod is None:
        return False
    # Direct file:  utils.py
    if os.path.isfile(os.path.join(root_dir, mod + extension)):
        return True
    # Package:  utils/__init__.py
    if os.path.isfile(os.path.join(root_dir, mod, "__init__" + extension)):
        return True
    # Also handle dotted paths for from-imports:  from utils.auth import X → utils/auth.py
    if line.strip().startswith("from "):
        rest = line.strip()[5:].split()[0]   # 'utils.auth'
        as_path = rest.replace(".", os.sep) + extension
        if os.path.isfile(os.path.join(root_dir, as_path)):
            return True
    return False


def extract_local_imports(prefix: str, root_dir: str) -> list[str]:
    """
    Resolve import statements in prefix to actual .py file paths in root_dir.
    Only returns paths that exist on disk — library imports are silently skipped.
    """
    resolved = []
    try:
        tree = ast.parse(prefix)
    except SyntaxError:
        return resolved
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            candidate = os.path.join(root_dir, node.module.replace(".", os.sep) + extension)
            if os.path.isfile(candidate):
                resolved.append(candidate)
            else:
                # also try package __init__
                pkg = os.path.join(root_dir, node.module.replace(".", os.sep), "__init__" + extension)
                if os.path.isfile(pkg):
                    resolved.append(pkg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                candidate = os.path.join(root_dir, alias.name.replace(".", os.sep) + extension)
                if os.path.isfile(candidate):
                    resolved.append(candidate)
    return list(set(resolved))


def filter_local_import_lines(lines: list[str], root_dir: str) -> list[str]:
    """
    From a list of import lines, keep only those that resolve to a file in root_dir.
    Library imports (torch, os, sklearn, …) are dropped.
    Non-import lines are passed through unchanged.
    """
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            if is_local_import(stripped, root_dir):
                result.append(line)
            # else: library import — silently drop
        else:
            result.append(line)
    return result


def get_prefix_for_submission(prefix: str, mode: str, root_dir: str, limit: int = 10) -> str:
    """
    Process the prefix written to the submission.
    - 'full'   → strip library imports only, keep everything else
    - 'simple' → last `limit` non-library-import lines
    - 'smart'  → local import lines + last `limit` code lines (deduped)

    Library imports (anything that doesn't resolve to a repo file) are always removed
    because they waste tokens and add noise without helping the model.
    """
    lines = prefix.splitlines()

    # Split into import lines and code lines
    import_lines = [l for l in lines if l.strip().startswith(("import ", "from "))]
    code_lines   = [l for l in lines if not l.strip().startswith(("import ", "from "))]

    # Keep only local imports
    local_imports = filter_local_import_lines(import_lines, root_dir)

    if mode == "full":
        # Rebuild: local imports at top, all code lines below
        return "\n".join(local_imports + code_lines)

    if mode == "simple":
        tail = code_lines[-limit:] if len(code_lines) > limit else code_lines
        return "\n".join(tail)

    # smart (default): local imports + tail of code lines
    tail    = code_lines[-limit:] if len(code_lines) > limit else code_lines
    combined = list(dict.fromkeys(local_imports + tail))
    return "\n".join(combined)


def get_retrieval_prefix(prefix: str, root_dir: str, limit: int = 20) -> str:
    """
    Build the query string used for file-level retrieval.
    - Local import lines only (library imports dropped — they don't help find repo files)
    - Plus last `limit` code lines (immediate context near the cursor)
    """
    lines        = prefix.splitlines()
    import_lines = [l for l in lines if l.strip().startswith(("import ", "from "))]
    code_lines   = [l for l in lines if not l.strip().startswith(("import ", "from "))]

    local_imports = filter_local_import_lines(import_lines, root_dir)
    tail          = code_lines[-limit:] if len(code_lines) > limit else code_lines
    combined      = list(dict.fromkeys(local_imports + tail))
    return "\n".join(combined)


def get_retrieval_suffix(suffix: str, mode: str) -> str:
    """Filter suffix for retrieval: drop closing-bracket-only suffixes when mode='filtered'."""
    if mode == "none":
        return ""
    if mode == "filtered" and re.match(r"^[\s}\)\]:,;]*$", suffix.strip()):
        return ""
    return suffix


def _indentation(line: str) -> int:
    """Number of leading spaces (tabs counted as 4)."""
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def _current_scope_from_prefix(lines: list[str]) -> tuple[int, int]:
    """
    Walk backwards from the cursor (last non-empty line) to find the
    opening line of the most meaningful enclosing syntactic block.

    Returns (scope_start_idx, cursor_idx) as indices into `lines`.

    Priority order when climbing:
      - Always stop at: def, async def, class, for, while, with
        (these define the real unit of work).
      - Keep climbing through: if, elif, else, try, except, finally
        (these are sub-blocks; we want the enclosing loop/function).
    This means the query captures e.g. the full `for` loop even when
    the cursor is inside a nested `if` inside that loop.
    """
    STRONG = ("def ", "async def ", "class ", "for ", "while ", "with ")

    if not lines:
        return 0, 0

    cursor_idx = len(lines) - 1
    while cursor_idx > 0 and not lines[cursor_idx].strip():
        cursor_idx -= 1

    cursor_indent = _indentation(lines[cursor_idx])
    scope_start   = 0   # fallback: top of file

    for i in range(cursor_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        indent = _indentation(lines[i])
        if indent < cursor_indent and stripped.endswith(":"):
            scope_start = i
            if any(stripped.startswith(kw) for kw in STRONG):
                break           # found a strong scope — stop here
            cursor_indent = indent  # weak scope (if/else/…) — keep climbing

    return scope_start, cursor_idx


def get_embedding_query(prefix: str, suffix: str,
                        context_lines_above: int = 8,
                        suffix_lines: int = 10) -> str:
    """
    Build the embedding query for chunk scoring using the current syntactic scope.

    Composition:
      1. `context_lines_above` lines immediately before the scope header
         (sibling functions, class body, outer context).
      2. Full current scope from its opening keyword (def/for/if/…) to cursor.
      3. Up to `suffix_lines` of filtered suffix (dropped if only brackets).

    Import lines are stripped — they add noise to semantic chunk matching.

    Examples:
      - Cursor inside a method  → captures: def header + body so far
      - Cursor inside a for loop inside a function → captures: for + body
      - Cursor at module level  → captures: last context_lines_above code lines
    """
    all_lines  = prefix.splitlines()
    code_lines = [l for l in all_lines
                  if not l.strip().startswith(("import ", "from "))]

    if not code_lines:
        clean = get_retrieval_suffix(suffix, "filtered")
        return "\n".join(clean.splitlines()[:suffix_lines])

    scope_start, cursor_idx = _current_scope_from_prefix(code_lines)

    # Lines above the scope header — outer context
    above_start = max(0, scope_start - context_lines_above)
    above_block = code_lines[above_start:scope_start]

    # Full current scope from opening keyword to cursor
    scope_block = code_lines[scope_start: cursor_idx + 1]

    # Suffix: first N meaningful lines, drop if just closing brackets
    clean_suffix = "\n".join(
        get_retrieval_suffix(suffix, "filtered").splitlines()[:suffix_lines]
    )

    parts = [b for b in [
        "\n".join(above_block),
        "\n".join(scope_block),
        clean_suffix,
    ] if b.strip()]

    return "\n".join(parts).strip()


def trim_suffix(suffix: str) -> str:
    lines = suffix.split("\n")
    return "\n".join(lines[:10]) if len(lines) > 10 else suffix


def estimate_tokens(text: str) -> int:
    return len(text) // 4


# ===========================================================================
# FILE COLLECTION HELPERS
# ===========================================================================

def all_repo_files(root_dir: str, min_lines: int = 10) -> list[str]:
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extension):
                fp = os.path.join(dirpath, filename)
                try:
                    with open(fp, encoding="utf-8") as f:
                        if sum(1 for _ in f) >= min_lines:
                            found.append(fp)
                except Exception:
                    pass
    return found


def collect_recent_files(root_dir: str, recent_filenames: list[str], min_lines: int = 10) -> list[str]:
    result = []
    for filename in recent_filenames:
        if not filename.endswith(extension):
            continue
        fp = os.path.join(root_dir, filename)
        try:
            with open(fp, encoding="utf-8") as f:
                if sum(1 for _ in f) >= min_lines:
                    result.append(fp)
        except Exception:
            pass
    return result


def chunk_file(content: str, chunk_size: int = 30) -> list[str]:
    """Overlapping windows of chunk_size lines (50% overlap)."""
    lines  = content.splitlines()
    chunks = []
    step   = max(1, chunk_size // 2)
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def assemble_context(ordered_files: list[tuple[str, str]], root_dir: str,
                     token_budget: int = 6000, most_relevant_last: bool = True) -> str:
    """
    Given a list of (file_path, content_excerpt) tuples already sorted by relevance
    (least → most relevant), assemble into a <|file_sep|>-delimited context string.

    most_relevant_last=True: most relevant file placed closest to the generation point
    (confirmed best by LCA paper — path distance puts closest file last).
    """
    parts = []
    tokens_used = 0
    for fp, content in ordered_files:
        cost = estimate_tokens(content)
        if tokens_used + cost > token_budget:
            # Try to include a truncated version of what remains
            remaining = (token_budget - tokens_used) * 4
            if remaining > 200:
                content = content[:remaining]
                cost    = estimate_tokens(content)
            else:
                continue
        clean_name = fp[len(root_dir) + 1:]
        parts.append(FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL, file_name=clean_name, file_content=content
        ))
        tokens_used += cost

    if most_relevant_last:
        # parts are in ascending relevance order already — no reorder needed
        pass

    return "".join(parts)


# ===========================================================================
# STRATEGY 1: PATH DISTANCE
# ---------------------------------------------------------------------------
# From LCA paper: files closer in the directory tree to the completion file
# are most relevant. Closer file goes LAST (nearest to generation point).
# Secondary sort by Lines IoU when path distances tie.
# This was the best context composer in the LCA perplexity experiments (+0.25
# drop vs file-level baseline).
# ===========================================================================

def path_distance(file_a: str, file_b: str) -> int:
    """
    Number of directory hops between two file paths.
    e.g. a/b/c.py and a/b/d/e.py → distance = 3 (up 1, down 2)
    """
    parts_a = file_a.replace("\\", "/").split("/")
    parts_b = file_b.replace("\\", "/").split("/")
    common  = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            common += 1
        else:
            break
    return (len(parts_a) - common) + (len(parts_b) - common)


def lines_iou(content_a: str, content_b: str, min_len: int = 5) -> float:
    """
    Intersection-over-Union of non-trivial lines shared between two files.
    Used as a secondary sort key when path distances are equal.
    From LCA pretraining paper — Lines IoU matches Path Distance performance.
    """
    def sig_lines(text: str) -> set[str]:
        return {l.strip() for l in text.splitlines()
                if len(l.strip()) >= min_len}
    sa, sb = sig_lines(content_a), sig_lines(content_b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def collect_path_distance_context(
    root_dir:         str,
    prefix:           str,
    suffix:           str,
    recent_filenames: list[str],
    max_files:        int  = 5,
    token_budget:     int  = 6000,
) -> str:
    """
    Rank ALL repo .py files by path distance from the completion file.
    - Completion file path is inferred from the prefix (last import or fallback).
    - Tie-break with Lines IoU against prefix+suffix.
    - Import-resolved files get a distance bonus (treated as distance=0).
    - Recent files get a small bonus too (distance capped at 1).
    - Files ordered least-relevant → most-relevant; most relevant goes last
      (closest to the generation point, as confirmed by LCA experiments).
    """
    all_files    = all_repo_files(root_dir)
    import_files = set(extract_local_imports(prefix, root_dir))
    recent_files = set(collect_recent_files(root_dir, recent_filenames))
    # Use library-filtered retrieval prefix so library names don't pollute IoU scoring
    query_text   = get_retrieval_prefix(prefix, root_dir) + "\n" + get_retrieval_suffix(suffix, "filtered")

    scored = []
    for fp in all_files:
        try:
            content = open(fp, encoding="utf-8").read()
        except Exception:
            continue

        # Compute path distance relative to root (simulate completion file location
        # using the deepest import-resolved file, or root itself)
        if import_files:
            ref = min(import_files, key=lambda p: path_distance(fp, p))
            dist = path_distance(fp, ref)
        else:
            # Estimate location from prefix indentation/package structure
            dist = fp.count(os.sep) - root_dir.count(os.sep)

        # Bonuses: imports are most trusted, recent files are next
        if fp in import_files:
            dist = 0
        elif fp in recent_files:
            dist = min(dist, 1)

        iou = lines_iou(query_text, content)
        scored.append((dist, -iou, fp, content))   # sort: dist asc, iou desc

    # Sort: closest first = most relevant → goes to the end of context
    scored.sort(key=lambda x: (x[0], x[1]))

    # Take top max_files, ordered least→most relevant for assembly
    top = scored[:max_files]
    # Reverse so least relevant is first in context, most relevant is last
    top_ordered = list(reversed(top))

    file_pairs = [(fp, content) for _, _, fp, content in top_ordered]
    return assemble_context(file_pairs, root_dir, token_budget, most_relevant_last=True)


# ===========================================================================
# STRATEGY 2: LINES IoU  (standalone)
# ---------------------------------------------------------------------------
# From LCA pretraining paper: ranking files purely by shared-line overlap with
# the completion context (prefix+suffix) performs on par with path distance,
# and sometimes better for the `Lines IoU .py` composer (51.8 inproject vs
# 48.8 for Path Distance in Or-16K eval mode).
# No file tree structure needed — purely content-based.
# ===========================================================================

def collect_lines_iou_context(
    root_dir:         str,
    prefix:           str,
    suffix:           str,
    recent_filenames: list[str],
    max_files:        int  = 5,
    token_budget:     int  = 6000,
) -> str:
    """
    Rank all repo .py files by Lines IoU with prefix+suffix.
    Import-resolved and recent files get a boost.
    Most relevant file (highest IoU) placed last in the context string.
    """
    all_files    = all_repo_files(root_dir)
    import_files = set(extract_local_imports(prefix, root_dir))
    recent_files = set(collect_recent_files(root_dir, recent_filenames))
    query_text   = get_retrieval_prefix(prefix, root_dir) + "\n" + get_retrieval_suffix(suffix, "filtered")

    scored = []
    for fp in all_files:
        try:
            content = open(fp, encoding="utf-8").read()
        except Exception:
            continue

        iou   = lines_iou(query_text, content)
        boost = 0.3 if fp in import_files else (0.1 if fp in recent_files else 0.0)
        scored.append((iou + boost, fp, content))

    scored.sort(key=lambda x: x[0])              # ascending: least relevant first
    top = scored[-max_files:]                     # last N = most relevant

    file_pairs = [(fp, content) for _, fp, content in top]
    # top is already least→most relevant order (ascending sort, last N)
    return assemble_context(file_pairs, root_dir, token_budget, most_relevant_last=True)


# ===========================================================================
# STRATEGY 3: BEST  (combined)
# ---------------------------------------------------------------------------
# Combines all signals:
#   1. Direct import resolution  (highest precision)
#   2. Path distance ranking     (structural proximity)
#   3. Lines IoU ranking         (content overlap)
#   4. Recent files              (developer working set)
#   5. Embedding chunk scoring   (semantic relevance for chunk selection)
#
# File selection: weighted rank fusion of path_distance + lines_iou + recency
# Chunk selection: embedding similarity against code-intent query (no imports)
# Ordering: least relevant → most relevant (most relevant file placed last)
# ===========================================================================

def rank_fusion(scores_list: list[list[tuple[float, str]]], weights: list[float]) -> dict[str, float]:
    """
    Combine multiple ranked lists into a single score per file path.
    Each scores_list[i] is a list of (score, file_path) already normalized 0..1.
    """
    combined: dict[str, float] = {}
    for score_list, w in zip(scores_list, weights):
        for score, fp in score_list:
            combined[fp] = combined.get(fp, 0.0) + w * score
    return combined


def collect_best_context(
    root_dir:         str,
    prefix:           str,
    suffix:           str,
    recent_filenames: list[str],
    chunk_size:       int  = 30,
    top_chunks:       int  = 3,
    max_files:        int  = 5,
    token_budget:     int  = 6000,
) -> str:
    """
    Full combined strategy:

    File ranking (rank fusion of three signals):
      - Path distance: closer in directory tree → higher score
      - Lines IoU: more shared lines with prefix+suffix → higher score
      - Recency: recently modified files get a direct boost

    Import-resolved files are pinned to the top (guaranteed included).

    Chunk selection inside each file:
      - Embedding similarity against code-intent query (prefix without imports + suffix)
      - Top-K chunks per file, globally ranked, greedily filling token budget

    Context assembly:
      - Ordered least-relevant → most-relevant
      - Most relevant file placed LAST (closest to generation point)
    """
    all_files    = all_repo_files(root_dir)
    import_files = set(extract_local_imports(prefix, root_dir))
    recent_set   = set(collect_recent_files(root_dir, recent_filenames))
    query_text   = get_retrieval_prefix(prefix, root_dir) + "\n" + get_retrieval_suffix(suffix, "filtered")

    if not all_files:
        return ""

    # ------------------------------------------------------------------
    # 1. Score every file — three priority tiers
    #
    #   Tier 3 — RECENT files (score=3.0): pinned as most relevant.
    #             Developer's active working set; always placed last
    #             in the context (closest to the generation point).
    #
    #   Tier 2 — IMPORT-resolved files (score=2.0): direct dependencies
    #             of the completion file; always included.
    #
    #   Tier 1 — everything else: scored by IoU × path_proximity
    #             path_proximity = 1 / (path_distance + 1)  ∈ (0, 1]
    #             Multiplying rewards files that are BOTH content-similar
    #             AND structurally close. A distant file with high IoU
    #             scores lower than a nearby file with the same IoU.
    # ------------------------------------------------------------------
    file_contents: dict[str, str] = {}
    fused:         dict[str, float] = {}

    for fp in all_files:
        try:
            content = open(fp, encoding="utf-8").read()
        except Exception:
            continue
        file_contents[fp] = content

        # Tier 3: recent — always most relevant
        if fp in recent_set:
            fused[fp] = 3.0
            continue

        # Tier 2: import-resolved — direct dependency
        if fp in import_files:
            fused[fp] = 2.0
            continue

        # Tier 1: IoU × path_proximity
        iou = lines_iou(query_text, content)

        if import_files:
            dist = min(path_distance(fp, ref) for ref in import_files)
        else:
            dist = fp.count(os.sep) - root_dir.count(os.sep)

        proximity = 1.0 / (dist + 1)      # 1.0 when dist=0, 0.5 at dist=1, …
        fused[fp] = iou * proximity

    # Sort ascending: least relevant first, most relevant last in context
    sorted_files   = sorted(fused.items(), key=lambda x: x[1])
    candidates     = [fp for fp, _ in sorted_files if fp in file_contents]
    candidates_top = candidates[-(max_files * 3):]

    # ------------------------------------------------------------------
    # 2. Embedding chunk scoring inside candidate files
    # ------------------------------------------------------------------
    emb_query = get_embedding_query(prefix, suffix)
    query_emb = embed_model.encode(emb_query, convert_to_tensor=True)

    # (chunk_score, fused_file_score, file_path, chunk_text)
    scored_chunks: list[tuple[float, float, str, str]] = []

    for fp in candidates_top:
        content = file_contents[fp]
        chunks  = chunk_file(content, chunk_size)
        if not chunks:
            scored_chunks.append((0.0, fused[fp], fp, content))
            continue

        chunk_embs = embed_model.encode(chunks, convert_to_tensor=True)
        sims       = util.cos_sim(query_emb, chunk_embs)[0]

        k = min(top_chunks, len(chunks))
        for idx in torch.topk(sims, k).indices.tolist():
            scored_chunks.append((sims[idx].item(), fused.get(fp, 0.0), fp, chunks[idx]))

    # ------------------------------------------------------------------
    # 3. Rank and greedily fill token budget
    # ------------------------------------------------------------------
    # Primary: chunk embedding score; secondary: file fused score
    scored_chunks.sort(key=lambda x: (x[0] + x[1]), reverse=True)

    selected: dict[str, list[str]] = {}
    file_order: list[str] = []          # insertion order tracks relevance
    tokens_used = 0

    for chunk_score, file_score, fp, chunk in scored_chunks:
        cost = estimate_tokens(chunk)
        if tokens_used + cost > token_budget:
            continue
        if fp not in selected:
            if len(selected) >= max_files:
                continue
            selected[fp] = []
            file_order.append(fp)
        selected[fp].append(chunk)
        tokens_used += cost

    if not selected:
        return ""

    # ------------------------------------------------------------------
    # 4. Assemble: order files by fused score ascending (least relevant first)
    #    so most relevant lands last — closest to the generation point.
    # ------------------------------------------------------------------
    file_order_sorted = sorted(file_order, key=lambda fp: fused.get(fp, 0.0))

    parts = []
    for fp in file_order_sorted:
        clean_name   = fp[len(root_dir) + 1:]
        file_excerpt = "\n...\n".join(selected[fp])
        parts.append(FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL, file_name=clean_name, file_content=file_excerpt
        ))
        print(f"  [best] {clean_name}  ({len(selected[fp])} chunk(s), "
              f"fused={fused.get(fp,0):.2f})")

    return "".join(parts)


# ===========================================================================
# LEGACY SINGLE-FILE STRATEGIES  (kept for baseline comparisons)
# ===========================================================================

def find_random_file(root_dir: str, min_lines: int = 10) -> str:
    files = all_repo_files(root_dir, min_lines)
    return random.choice(files) if files else None


def find_bm25_file(root_dir: str, prefix: str, suffix: str, min_lines: int = 10) -> str:
    def tokenize(s: str) -> list[str]:
        return "".join(c if (c.isalnum() or c == "_") else " " for c in s.lower()).split()
    corpus, file_names = [], []
    for fp in all_repo_files(root_dir, min_lines):
        try:
            corpus.append(tokenize(open(fp, encoding="utf-8").read()))
            file_names.append(fp)
        except Exception:
            pass
    if not file_names:
        return None
    return file_names[BM25Okapi(corpus).get_scores(tokenize(prefix + " " + suffix)).argmax()]


def find_embedding_file(root_dir: str, query_text: str, min_lines: int = 10) -> str:
    file_names, corpus_contents = [], []
    for fp in all_repo_files(root_dir, min_lines):
        try:
            corpus_contents.append(open(fp, encoding="utf-8").read())
            file_names.append(fp)
        except Exception:
            pass
    if not file_names:
        return None
    q_emb = embed_model.encode(query_text, convert_to_tensor=True)
    c_emb = embed_model.encode(corpus_contents, convert_to_tensor=True)
    return file_names[torch.argmax(util.cos_sim(q_emb, c_emb)[0]).item()]


def collect_smart_context(root_dir, prefix, suffix, recent_filenames,
                          chunk_size=30, top_chunks=3, max_files=4, token_budget=6000):
    """Original smart_multi strategy — kept for comparison."""
    import_files = extract_local_imports(prefix, root_dir)
    recent_files = collect_recent_files(root_dir, recent_filenames)
    seen: set[str] = set()
    candidates: list[str] = []
    for fp in import_files + recent_files:
        if fp not in seen:
            seen.add(fp)
            candidates.append(fp)
    if len(candidates) < 2:
        for fp in all_repo_files(root_dir):
            if fp not in seen:
                seen.add(fp)
                candidates.append(fp)
    if not candidates:
        return ""
    emb_query = get_embedding_query(prefix, suffix)
    query_emb = embed_model.encode(emb_query, convert_to_tensor=True)
    scored_chunks: list[tuple[float, str, str]] = []
    for fp in candidates[:max_files * 4]:
        try:
            content = open(fp, encoding="utf-8").read()
        except Exception:
            continue
        chunks = chunk_file(content, chunk_size)
        if not chunks:
            continue
        sims = util.cos_sim(query_emb, embed_model.encode(chunks, convert_to_tensor=True))[0]
        for idx in torch.topk(sims, min(top_chunks, len(chunks))).indices.tolist():
            scored_chunks.append((sims[idx].item(), fp, chunks[idx]))
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    selected: dict[str, list[str]] = {}
    tokens_used = 0
    for score, fp, chunk in scored_chunks:
        cost = estimate_tokens(chunk)
        if tokens_used + cost > token_budget:
            continue
        if fp not in selected:
            if len(selected) >= max_files:
                continue
            selected[fp] = []
        selected[fp].append(chunk)
        tokens_used += cost
    parts = []
    for fp, chunks in selected.items():
        clean_name = fp[len(root_dir) + 1:]
        parts.append(FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL, file_name=clean_name,
            file_content="\n...\n".join(chunks)
        ))
        print(f"  [smart_multi] {clean_name} ({len(chunks)} chunk(s))")
    return "".join(parts)


# ===========================================================================
# MAIN LOOP
# ===========================================================================

completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")
prediction_file_name   = f"{language}-{stage}-{strategy}-p_{args.prefix_mode}-s_{args.suffix_mode}"
if args.trim_prefix:
    prediction_file_name += "-short-prefix"
if args.trim_suffix:
    prediction_file_name += "-short-suffix"
predictions_file = os.path.join("predictions", f"{prediction_file_name}.jsonl")

os.makedirs("predictions", exist_ok=True)

with jsonlines.open(completion_points_file, 'r') as reader:
    with jsonlines.open(predictions_file, 'w') as writer:
        for datapoint in reader:
            repo_path      = datapoint['repo'].replace("/", "__")
            repo_revision  = datapoint['revision']
            root_directory = os.path.join(
                "data", f"repositories-{language}-{stage}", f"{repo_path}-{repo_revision}"
            )
            raw_prefix = datapoint['prefix']
            raw_suffix = datapoint['suffix']
            recent     = datapoint.get('modified', [])

            # ----------------------------------------------------------
            if strategy == "random":
                fn      = find_random_file(root_directory)
                content = open(fn, encoding="utf-8").read()
                name    = fn[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)
                print(f"Picked: {name}")

            elif strategy == "bm25":
                fn      = find_bm25_file(root_directory,
                                         get_retrieval_prefix(raw_prefix, root_directory),
                                         get_retrieval_suffix(raw_suffix, args.suffix_mode))
                content = open(fn, encoding="utf-8").read()
                name    = fn[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)
                print(f"Picked: {name}")

            elif strategy == "recent":
                rec = collect_recent_files(root_directory, recent)
                fn  = rec[0] if rec else find_random_file(root_directory)
                content = open(fn, encoding="utf-8").read()
                name    = fn[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)
                print(f"Picked: {name}")

            elif strategy == "embeddings":
                fn = find_embedding_file(
                    root_directory,
                    get_retrieval_prefix(raw_prefix, root_directory) + "\n" +
                    get_retrieval_suffix(raw_suffix, args.suffix_mode))
                content = open(fn, encoding="utf-8").read()
                name    = fn[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)
                print(f"Picked: {name}")

            elif strategy == "smart_multi":
                context = collect_smart_context(
                    root_directory, raw_prefix, raw_suffix, recent,
                    args.chunk_lines, args.top_chunks, args.max_files, args.token_budget)
                if not context:
                    fn      = find_random_file(root_directory)
                    content = open(fn, encoding="utf-8").read()
                    name    = fn[len(root_directory) + 1:]
                    context = FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)
                print(f"[smart_multi] ~{estimate_tokens(context)} tokens")

            elif strategy == "path_distance":
                context = collect_path_distance_context(
                    root_directory, raw_prefix, raw_suffix, recent,
                    args.max_files, args.token_budget)
                if not context:
                    fn      = find_random_file(root_directory)
                    content = open(fn, encoding="utf-8").read()
                    name    = fn[len(root_directory) + 1:]
                    context = FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)
                print(f"[path_distance] ~{estimate_tokens(context)} tokens")

            elif strategy == "lines_iou":
                context = collect_lines_iou_context(
                    root_directory, raw_prefix, raw_suffix, recent,
                    args.max_files, args.token_budget)
                if not context:
                    fn      = find_random_file(root_directory)
                    content = open(fn, encoding="utf-8").read()
                    name    = fn[len(root_directory) + 1:]
                    context = FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)
                print(f"[lines_iou] ~{estimate_tokens(context)} tokens")

            elif strategy == "best":
                context = collect_best_context(
                    root_directory, raw_prefix, raw_suffix, recent,
                    args.chunk_lines, args.top_chunks, args.max_files, args.token_budget)
                if not context:
                    fn      = find_random_file(root_directory)
                    content = open(fn, encoding="utf-8").read()
                    name    = fn[len(root_directory) + 1:]
                    context = FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)
                print(f"[best] ~{estimate_tokens(context)} tokens")

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            submission = {"context": context}
            if args.trim_prefix:
                submission["prefix"] = get_prefix_for_submission(raw_prefix, "smart", root_directory)
            if args.trim_suffix:
                submission["suffix"] = trim_suffix(raw_suffix)
            writer.write(submission)