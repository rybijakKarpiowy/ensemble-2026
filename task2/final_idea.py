import os
import re
import ast
import jsonlines
import argparse

from sentence_transformers import SentenceTransformer, util
import torch


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

argparser = argparse.ArgumentParser()
argparser.add_argument("--stage",        type=str, default="practice", help="Stage of the project")
argparser.add_argument("--lang",         type=str, default="python",   help="Language")
argparser.add_argument("--chunk-lines",  type=int, default=30,   help="Lines per chunk for chunk-based strategies")
argparser.add_argument("--top-chunks",   type=int, default=3,    help="Top-K chunks to keep per file")
argparser.add_argument("--max-files",    type=int, default=5,    help="Max files to include in context")
argparser.add_argument("--token-budget", type=int, default=6000, help="Approximate token budget for context")

args = argparser.parse_args()

stage    = args.stage
language = args.lang

if language == "python":
    extension = ".py"
elif language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

print(f"Running 'best' strategy  |  stage='{stage}'  |  lang='{language}'")

FILE_SEP_SYMBOL     = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# ===========================================================================
# PREFIX / SUFFIX UTILITIES
# ===========================================================================

def _module_name_from_line(line: str) -> str | None:
    """
    Extract the top-level module name from an import line.
      'import os'                -> 'os'
      'from utils.auth import X' -> 'utils'   (first component)
      'import torch, numpy'      -> 'torch'   (first only -- good enough for lookup)
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
    if os.path.isfile(os.path.join(root_dir, mod + extension)):
        return True
    if os.path.isfile(os.path.join(root_dir, mod, "__init__" + extension)):
        return True
    if line.strip().startswith("from "):
        rest = line.strip()[5:].split()[0]
        as_path = rest.replace(".", os.sep) + extension
        if os.path.isfile(os.path.join(root_dir, as_path)):
            return True
    return False


def extract_local_imports(prefix: str, root_dir: str) -> list[str]:
    """
    Resolve import statements in prefix to actual file paths in root_dir.
    Only returns paths that exist on disk -- library imports are silently skipped.
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
    Library imports (torch, os, sklearn, ...) are dropped.
    Non-import lines are passed through unchanged.
    """
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            if is_local_import(stripped, root_dir):
                result.append(line)
        else:
            result.append(line)
    return result


def get_retrieval_prefix(prefix: str, root_dir: str, limit: int = 20) -> str:
    """
    Build the query string used for file-level retrieval.
    - Local import lines only (library imports dropped)
    - Plus last `limit` code lines (immediate context near the cursor)
    """
    lines        = prefix.splitlines()
    import_lines = [l for l in lines if l.strip().startswith(("import ", "from "))]
    code_lines   = [l for l in lines if not l.strip().startswith(("import ", "from "))]

    local_imports = filter_local_import_lines(import_lines, root_dir)
    tail          = code_lines[-limit:] if len(code_lines) > limit else code_lines
    combined      = list(dict.fromkeys(local_imports + tail))
    return "\n".join(combined)


def get_retrieval_suffix(suffix: str) -> str:
    """Drop closing-bracket-only suffixes -- they add noise without helping retrieval."""
    if re.match(r"^[\s}\)\]:,;]*$", suffix.strip()):
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
      - Keep climbing through: if, elif, else, try, except, finally
    """
    STRONG = ("def ", "async def ", "class ", "for ", "while ", "with ")

    if not lines:
        return 0, 0

    cursor_idx = len(lines) - 1
    while cursor_idx > 0 and not lines[cursor_idx].strip():
        cursor_idx -= 1

    cursor_indent = _indentation(lines[cursor_idx])
    scope_start   = 0

    for i in range(cursor_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        indent = _indentation(lines[i])
        if indent < cursor_indent and stripped.endswith(":"):
            scope_start = i
            if any(stripped.startswith(kw) for kw in STRONG):
                break
            cursor_indent = indent

    return scope_start, cursor_idx


def get_embedding_query(prefix: str, suffix: str,
                        context_lines_above: int = 8,
                        suffix_lines: int = 10) -> str:
    """
    Build the embedding query for chunk scoring using the current syntactic scope.

    Composition:
      1. `context_lines_above` lines immediately before the scope header.
      2. Full current scope from its opening keyword to cursor.
      3. Up to `suffix_lines` of filtered suffix.

    Import lines are stripped -- they add noise to semantic chunk matching.
    """
    all_lines  = prefix.splitlines()
    code_lines = [l for l in all_lines
                  if not l.strip().startswith(("import ", "from "))]

    if not code_lines:
        clean = get_retrieval_suffix(suffix)
        return "\n".join(clean.splitlines()[:suffix_lines])

    scope_start, cursor_idx = _current_scope_from_prefix(code_lines)

    above_start  = max(0, scope_start - context_lines_above)
    above_block  = code_lines[above_start:scope_start]
    scope_block  = code_lines[scope_start: cursor_idx + 1]
    clean_suffix = "\n".join(get_retrieval_suffix(suffix).splitlines()[:suffix_lines])

    parts = [b for b in [
        "\n".join(above_block),
        "\n".join(scope_block),
        clean_suffix,
    ] if b.strip()]

    return "\n".join(parts).strip()


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def get_prefix_for_submission(prefix: str, root_dir: str) -> str:
    """
    Build the prefix to submit alongside the context.

    Returns: local import lines + the entire enclosing top-level block up to
    the cursor.

    "Top-level block" means everything from the nearest zero-indentation
    def/async def/class line above the cursor down to the cursor. This gives
    the model the full function/class signature and all the code written so
    far, which is the most informative slice of the prefix.

    If the cursor is already at module level (no enclosing def/class found),
    the last 20 non-import lines are returned instead.
    """
    lines = prefix.splitlines()

    # Collect local import lines from the entire prefix
    import_lines  = [l for l in lines if l.strip().startswith(("import ", "from "))]
    local_imports = filter_local_import_lines(import_lines, root_dir)

    # Non-import lines only for scope search
    code_lines = [l for l in lines if not l.strip().startswith(("import ", "from "))]

    if not code_lines:
        return "\n".join(local_imports)

    # Walk backwards to find the nearest zero-indent def/async def/class
    TOP_LEVEL = ("def ", "async def ", "class ")
    block_start = None
    for i in range(len(code_lines) - 1, -1, -1):
        stripped = code_lines[i].strip()
        if not stripped:
            continue
        if _indentation(code_lines[i]) == 0 and any(stripped.startswith(kw) for kw in TOP_LEVEL):
            block_start = i
            break

    if block_start is not None:
        block = code_lines[block_start:]
    else:
        # Module-level cursor: fall back to last 20 code lines
        block = code_lines[-20:]

    combined = list(dict.fromkeys(local_imports + block))
    return "\n".join(combined)


def get_suffix_for_submission(suffix: str, limit: int = 20) -> str:
    """
    Keep the first `limit` lines of the suffix, skipping any leading lines
    that contain only closing brackets / whitespace (noise without signal).
    """
    lines = suffix.splitlines()
    # Find the first line that has real content
    start = 0
    for i, line in enumerate(lines):
        if line.strip() and not re.match(r"^[\s}\)\]:,;]*$", line):
            start = i
            break
    lines = lines[start:start + limit]
    return "\n".join(lines)


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


# ===========================================================================
# SCORING PRIMITIVES
# ===========================================================================

def path_distance(file_a: str, file_b: str) -> int:
    """
    Number of directory hops between two file paths.
    e.g. a/b/c.py and a/b/d/e.py -> distance = 3 (up 1, down 2)
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


def lines_iou(content_a: str, content_b: str, min_len: int = 10) -> float:
    """
    Intersection-over-Union of non-trivial lines shared between two files.
    From LCA pretraining paper -- Lines IoU matches Path Distance performance.
    """
    def sig_lines(text: str) -> set[str]:
        return {l.strip() for l in text.splitlines()
                if len(l.strip()) >= min_len}
    sa, sb = sig_lines(content_a), sig_lines(content_b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ===========================================================================
# BEST STRATEGY
# ---------------------------------------------------------------------------
# Combines all signals:
#   1. Direct import resolution  (highest precision)
#   2. Path distance ranking     (structural proximity)
#   3. Lines IoU ranking         (content overlap)
#   4. Recent files              (developer working set)
#   5. Embedding chunk scoring   (semantic relevance for chunk selection)
#
# File selection: three priority tiers (recent > imports > IoU x proximity)
# Chunk selection: embedding similarity against code-intent query (no imports)
# Ordering: least relevant -> most relevant (most relevant file placed last)
# ===========================================================================

def collect_best_context(
    root_dir:         str,
    prefix:           str,
    suffix:           str,
    recent_filenames: list[str],
    chunk_size:       int  = 30,
    top_chunks:       int  = 4,
    max_files:        int  = 8,
    token_budget:     int  = 16000,
    completion_file:  str  = None,
) -> str:
    """
    Full combined strategy:

    File ranking (three priority tiers):
      - Tier 3 -- Recent files: pinned as most relevant (score=3.0)
      - Tier 2 -- Import-resolved files: direct dependencies (score=2.0)
      - Tier 1 -- Everything else: scored by IoU x path_proximity
                  path_proximity = 1 / (path_distance + 1) in (0, 1]

    Chunk selection inside each file:
      - Embedding similarity against code-intent query (prefix without imports + suffix)
      - Top-K chunks per file, globally ranked, greedily filling token budget

    Context assembly:
      - Ordered least-relevant -> most-relevant
      - Most relevant file placed LAST (closest to generation point)
    """
    all_files    = all_repo_files(root_dir)
    import_files = set(extract_local_imports(prefix, root_dir))
    recent_set   = set(collect_recent_files(root_dir, recent_filenames))
    query_text   = get_retrieval_prefix(prefix, root_dir) + "\n" + get_retrieval_suffix(suffix)

    if not all_files:
        return ""

    # ------------------------------------------------------------------
    # 1. Score every file -- three priority tiers
    # ------------------------------------------------------------------
    file_contents: dict[str, str] = {}
    fused:         dict[str, float] = {}

    for fp in all_files:
        try:
            content = open(fp, encoding="utf-8").read()
        except Exception:
            continue
        file_contents[fp] = content

        if fp in recent_set:
            fused[fp] = 1.0
            continue

        if fp in import_files:
            fused[fp] = 1.0
            continue

        iou = lines_iou(query_text, content)

        if completion_file:
            dist = path_distance(fp, completion_file)
        elif import_files:
            dist = min(path_distance(fp, ref) for ref in import_files)
        else:
            dist = fp.count(os.sep) - root_dir.count(os.sep)

        proximity = 1.0 / (dist + 1)
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
    scored_chunks.sort(key=lambda x: (x[0] + x[1]), reverse=True)

    selected: dict[str, list[str]] = {}
    file_order: list[str] = []
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
    # 4. Assemble: least relevant first, most relevant last
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
# MAIN LOOP
# ===========================================================================

completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")
predictions_file       = os.path.join("predictions", f"{language}-{stage}-best.jsonl")

os.makedirs("predictions", exist_ok=True)

with jsonlines.open(completion_points_file, 'r') as reader:
    with jsonlines.open(predictions_file, 'w') as writer:
        for datapoint in reader:
            repo_path      = datapoint['repo'].replace("/", "__")
            repo_revision  = datapoint['revision']
            root_directory = os.path.join(
                "data", f"repositories-{language}-{stage}", f"{repo_path}-{repo_revision}"
            )
            raw_prefix      = datapoint['prefix']
            raw_suffix      = datapoint['suffix']
            recent          = datapoint.get('modified', [])
            completion_file = os.path.join(root_directory, os.path.basename(datapoint['path']))

            context = collect_best_context(
                root_directory, raw_prefix, raw_suffix, recent,
                args.chunk_lines, args.top_chunks, args.max_files, args.token_budget,
                completion_file)

            if not context:
                # Fallback: pick the file closest by path distance to the completion file
                all_files = all_repo_files(root_directory)
                if all_files:
                    fn      = min(all_files, key=lambda fp: path_distance(fp, completion_file))
                    content = open(fn, encoding="utf-8").read()
                    name    = fn[len(root_directory) + 1:]
                    context = FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL, file_name=name, file_content=content)

            print(f"[best] ~{estimate_tokens(context)} tokens")

            writer.write({
                "context": context,
                "prefix":  get_prefix_for_submission(raw_prefix, root_directory),
                "suffix":  get_suffix_for_submission(raw_suffix),
            })