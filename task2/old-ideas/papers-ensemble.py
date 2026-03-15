import os
import re
import ast
import warnings
import jsonlines
import argparse

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

argparser = argparse.ArgumentParser()
argparser.add_argument("--stage",        type=str, default="practice")
argparser.add_argument("--lang",         type=str, default="python")
argparser.add_argument("--chunk-lines",  type=int, default=40)
argparser.add_argument("--top-chunks",   type=int, default=3)
argparser.add_argument("--max-files",    type=int, default=5)
argparser.add_argument("--token-budget", type=int, default=14000,
                       help="Fill to 16K window; evaluator left-trims for Mellum (8K)")
argparser.add_argument("--embed-model",  type=str,
                       default="jinaai/jina-embeddings-v2-base-code",
                       help="Use 'all-MiniLM-L6-v2' as a faster fallback")

args = argparser.parse_args()

stage    = args.stage
language = args.lang

if language == "python":
    extension = ".py"
elif language == "kotlin":
    extension = ".kt"
else:
    raise ValueError(f"Unsupported language: {language}")

print(f"stage='{stage}' | lang='{language}' | budget={args.token_budget}")

FILE_SEP_SYMBOL    = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

# Code-specific embedding model; falls back to MiniLM if unavailable
try:
    embed_model = SentenceTransformer(args.embed_model, trust_remote_code=True)
    print(f"Loaded: {args.embed_model}")
except Exception as e:
    print(f"Falling back to all-MiniLM-L6-v2 ({e})")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# ===========================================================================
# TOKEN ESTIMATE
# ===========================================================================

def estimate_tokens(text: str) -> int:
    # ~3.5 chars/token for code (more punctuation than prose)
    return int(len(text) / 3.5)


# ===========================================================================
# IMPORT UTILITIES
# ===========================================================================

def _module_name_from_line(line: str) -> str | None:
    line = line.strip()
    if line.startswith("from "):
        parts = line[5:].split()
        return parts[0].split(".")[0] if parts else None
    if line.startswith("import "):
        parts = line[7:].split()
        return parts[0].split(".")[0].rstrip(",") if parts else None
    return None


def is_local_import(line: str, root_dir: str) -> bool:
    mod = _module_name_from_line(line)
    if not mod:
        return False
    if os.path.isfile(os.path.join(root_dir, mod + extension)):
        return True
    if os.path.isfile(os.path.join(root_dir, mod, "__init__" + extension)):
        return True
    if line.strip().startswith("from "):
        rest    = line.strip()[5:].split()[0]
        as_path = rest.replace(".", os.sep) + extension
        if os.path.isfile(os.path.join(root_dir, as_path)):
            return True
    return False


def extract_local_imports(prefix: str, root_dir: str) -> list[str]:
    """Resolve import statements to existing .py paths; discard library imports."""
    resolved = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(prefix)
    except SyntaxError:
        return resolved
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            c = os.path.join(root_dir, node.module.replace(".", os.sep) + extension)
            if os.path.isfile(c):
                resolved.append(c)
            else:
                pkg = os.path.join(root_dir, node.module.replace(".", os.sep),
                                   "__init__" + extension)
                if os.path.isfile(pkg):
                    resolved.append(pkg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                c = os.path.join(root_dir, alias.name.replace(".", os.sep) + extension)
                if os.path.isfile(c):
                    resolved.append(c)
    return list(set(resolved))


def _local_import_lines(lines: list[str], root_dir: str) -> list[str]:
    """Keep only import lines that resolve to a local repo file."""
    return [l for l in lines
            if l.strip().startswith(("import ", "from "))
            and is_local_import(l.strip(), root_dir)]


# ===========================================================================
# PREFIX / SUFFIX FOR SUBMISSION
# Prefix: local imports + last 10 code lines (library imports stripped)
# Suffix: first 10 lines (dropped if only closing brackets)
# ===========================================================================

def build_submission_prefix(prefix: str, root_dir: str, limit: int = 10) -> str:
    lines        = prefix.splitlines()
    import_lines = [l for l in lines if l.strip().startswith(("import ", "from "))]
    code_lines   = [l for l in lines if not l.strip().startswith(("import ", "from "))]
    local_imps   = _local_import_lines(import_lines, root_dir)
    tail         = code_lines[-limit:] if len(code_lines) > limit else code_lines
    return "\n".join(list(dict.fromkeys(local_imps + tail)))


def build_submission_suffix(suffix: str, limit: int = 10) -> str:
    lines = suffix.splitlines()
    return "\n".join(lines[:limit]) if len(lines) > limit else suffix


# ===========================================================================
# RETRIEVAL QUERY BUILDERS
# Used for BM25 file scoring — not for embeddings
# ===========================================================================

def _retrieval_query(prefix: str, suffix: str, root_dir: str,
                     code_limit: int = 20) -> str:
    """Local imports + last code_limit non-import lines + filtered suffix."""
    lines        = prefix.splitlines()
    import_lines = [l for l in lines if l.strip().startswith(("import ", "from "))]
    code_lines   = [l for l in lines if not l.strip().startswith(("import ", "from "))]
    local_imps   = _local_import_lines(import_lines, root_dir)
    tail         = code_lines[-code_limit:] if len(code_lines) > code_limit else code_lines

    # Drop suffix if it's only closing brackets — zero signal
    clean_suffix = suffix.strip()
    if re.match(r"^[\s}\)\]:,;]*$", clean_suffix):
        clean_suffix = ""

    parts = list(dict.fromkeys(local_imps + tail))
    if clean_suffix:
        parts.append(clean_suffix)
    return "\n".join(parts)


# ===========================================================================
# SCOPE-AWARE EMBEDDING QUERY
# Captures the current syntactic scope (def/for/while/with/class) + context
# above it + filtered suffix. Used for chunk-level embedding scoring.
# ===========================================================================

def _indentation(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def _scope_start(code_lines: list[str]) -> tuple[int, int]:
    """
    Walk backwards from cursor to find the enclosing strong-scope header.
    Strong: def, async def, class, for, while, with  → stop.
    Weak:   if, elif, else, try, except, finally     → keep climbing.
    Returns (scope_start_idx, cursor_idx).
    """
    STRONG = ("def ", "async def ", "class ", "for ", "while ", "with ")
    if not code_lines:
        return 0, 0
    cursor = len(code_lines) - 1
    while cursor > 0 and not code_lines[cursor].strip():
        cursor -= 1
    cur_indent  = _indentation(code_lines[cursor])
    scope_start = 0
    for i in range(cursor - 1, -1, -1):
        s = code_lines[i].strip()
        if not s:
            continue
        ind = _indentation(code_lines[i])
        if ind < cur_indent and s.endswith(":"):
            scope_start = i
            if any(s.startswith(kw) for kw in STRONG):
                break
            cur_indent = ind
    return scope_start, cursor


def build_embedding_query(prefix: str, suffix: str,
                          above_lines: int = 8,
                          suffix_lines: int = 10) -> str:
    """
    Embedding query composition:
      1. `above_lines` lines before the current scope header (outer context)
      2. Full current scope from its opening keyword to the cursor
      3. Up to `suffix_lines` of suffix (dropped if only brackets)
    All import lines stripped.
    """
    code_lines = [l for l in prefix.splitlines()
                  if not l.strip().startswith(("import ", "from "))]

    # Filtered suffix
    clean_suf = suffix.strip()
    if re.match(r"^[\s}\)\]:,;]*$", clean_suf):
        clean_suf = ""
    suf_block = "\n".join(suffix.splitlines()[:suffix_lines]) if clean_suf else ""

    if not code_lines:
        return suf_block

    ss, cursor = _scope_start(code_lines)
    above      = code_lines[max(0, ss - above_lines): ss]
    scope      = code_lines[ss: cursor + 1]

    parts = [b for b in [
        "\n".join(above),
        "\n".join(scope),
        suf_block,
    ] if b.strip()]
    return "\n".join(parts).strip()


# ===========================================================================
# BM25 FILE SCORING
# ===========================================================================

def _tok(text: str) -> list[str]:
    return "".join(c if (c.isalnum() or c == "_") else " "
                   for c in text.lower()).split()


def bm25_scores(query: str, file_contents: dict[str, str]) -> dict[str, float]:
    fps    = list(file_contents)
    corpus = [_tok(file_contents[fp]) for fp in fps]
    q_toks = _tok(query)
    if not corpus or not q_toks:
        return {fp: 0.0 for fp in fps}
    raw = BM25Okapi(corpus).get_scores(q_toks)
    mx  = raw.max() or 1.0
    return {fps[i]: float(raw[i]) / mx for i in range(len(fps))}


# ===========================================================================
# PATH DISTANCE + ANCHOR INFERENCE
# ===========================================================================

def _path_dist(a: str, b: str) -> int:
    pa = a.replace("\\", "/").split("/")
    pb = b.replace("\\", "/").split("/")
    c  = sum(1 for x, y in zip(pa, pb) if x == y)
    # find longest common prefix length
    c = 0
    for x, y in zip(pa, pb):
        if x == y:
            c += 1
        else:
            break
    return (len(pa) - c) + (len(pb) - c)


def _infer_anchor(prefix: str, root_dir: str,
                  import_files: set[str]) -> str | None:
    """
    Try to find the actual path of the file being completed.
    1. Look for a path hint comment in the first 5 lines.
    2. Use the centroid of import-resolved files (minimises avg distance).
    """
    for line in prefix.splitlines()[:5]:
        s = line.strip()
        if s.startswith("#") and (".py" in s or "path:" in s):
            for tok in s.split():
                if tok.endswith(".py") and not tok.startswith("#"):
                    c = os.path.join(root_dir, tok.lstrip("/"))
                    if os.path.isfile(c):
                        return c
    if not import_files:
        return None
    imp = list(import_files)
    if len(imp) == 1:
        return imp[0]
    best, best_avg = imp[0], float("inf")
    for ref in imp:
        avg = sum(_path_dist(ref, o) for o in imp if o != ref) / (len(imp) - 1)
        if avg < best_avg:
            best, best_avg = ref, avg
    return best


# ===========================================================================
# FUNCTION-BOUNDARY CHUNKING
# ===========================================================================

def _chunk_by_functions(content: str, max_lines: int = 60,
                         min_lines: int = 5) -> list[str]:
    lines      = content.splitlines()
    boundaries = [0]
    for i, line in enumerate(lines[1:], 1):
        s = line.strip()
        if (s.startswith(("def ", "async def ", "class ")) or
                (s.startswith("@") and i + 1 < len(lines) and
                 lines[i + 1].strip().startswith(("def ", "async def ", "class ")))):
            if i - boundaries[-1] >= min_lines:
                boundaries.append(i)
    if len(boundaries) <= 1:
        return _chunk_sliding(content, max_lines)
    boundaries.append(len(lines))
    chunks = []
    for start, end in zip(boundaries, boundaries[1:]):
        seg = lines[start:end]
        if len(seg) < min_lines:
            if chunks:
                chunks[-1] += "\n" + "\n".join(seg)
            continue
        if len(seg) > max_lines:
            for j in range(0, len(seg), max_lines):
                sub = "\n".join(seg[j: j + max_lines])
                if sub.strip():
                    chunks.append(sub)
        else:
            c = "\n".join(seg)
            if c.strip():
                chunks.append(c)
    return chunks


def _chunk_sliding(content: str, chunk_size: int = 40) -> list[str]:
    lines = content.splitlines()
    step  = max(1, chunk_size // 2)
    return ["\n".join(lines[i: i + chunk_size])
            for i in range(0, len(lines), step)
            if "\n".join(lines[i: i + chunk_size]).strip()]


def chunk_file(content: str, max_lines: int = 40) -> list[str]:
    return _chunk_by_functions(content, max_lines)


# ===========================================================================
# CHUNK DEDUPLICATION
# ===========================================================================

def dedup_chunks(chunks: list[str], threshold: float = 0.6) -> list[str]:
    kept, kept_sets = [], []
    for chunk in chunks:
        cset = {l.strip() for l in chunk.splitlines() if l.strip()}
        if not any(len(cset & p) / len(cset | p) >= threshold
                   for p in kept_sets if cset | p):
            kept.append(chunk)
            kept_sets.append(cset)
    return kept


# ===========================================================================
# FILE COLLECTION
# ===========================================================================

def repo_files(root_dir: str, min_lines: int = 10) -> list[str]:
    found = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(extension):
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, encoding="utf-8") as f:
                        if sum(1 for _ in f) >= min_lines:
                            found.append(fp)
                except Exception:
                    pass
    return found


def recent_files(root_dir: str, modified: list[str],
                 min_lines: int = 10) -> list[str]:
    result = []
    for fn in modified:
        if not fn.endswith(extension):
            continue
        fp = os.path.join(root_dir, fn)
        try:
            with open(fp, encoding="utf-8") as f:
                if sum(1 for _ in f) >= min_lines:
                    result.append(fp)
        except Exception:
            pass
    return result


# ===========================================================================
# CONTEXT ASSEMBLY
# ===========================================================================

def assemble(ordered_files: list[tuple[str, str]], root_dir: str,
             token_budget: int) -> str:
    """
    ordered_files: (path, content) sorted least→most relevant.
    Most relevant file placed last — closest to generation point,
    survives left-trimming by the evaluator on small windows (Mellum 8K).
    """
    parts, used = [], 0
    for fp, content in ordered_files:
        cost = estimate_tokens(content)
        if used + cost > token_budget:
            remaining = int((token_budget - used) * 3.5)
            if remaining > 200:
                content = content[:remaining]
            else:
                continue
        parts.append(FILE_COMPOSE_FORMAT.format(
            file_sep=FILE_SEP_SYMBOL,
            file_name=fp[len(root_dir) + 1:],
            file_content=content,
        ))
        used += estimate_tokens(content)
    return "".join(parts)


# ===========================================================================
# BEST CONTEXT PIPELINE
# ===========================================================================

def collect_best_context(
    root_dir:    str,
    prefix:      str,
    suffix:      str,
    modified:    list[str],
    chunk_lines: int = 40,
    top_chunks:  int = 3,
    max_files:   int = 5,
    budget:      int = 14000,
) -> str:
    """
    File scoring — three tiers:
      Tier 3 (3.0 + bm25×0.5): recently modified files, sub-ranked by BM25
      Tier 2 (2.0):             import-resolved files (direct dependencies)
      Tier 1 (0–1):             all others — BM25(query, file) × path_proximity
                                path_proximity = 1/(dist+1), dist from inferred anchor

    Chunk scoring:
      Function-boundary chunking → embedded with code model → top-K per file
      Scores normalised before combining (60% chunk + 40% file)
      Overlapping chunks deduplicated

    Assembly:
      Least relevant → most relevant order
      Most relevant file last (survives left-trim on Mellum 8K)
    """
    all_files    = repo_files(root_dir)
    import_files = set(extract_local_imports(prefix, root_dir))
    recent       = set(recent_files(root_dir, modified))

    if not all_files:
        return ""

    query    = _retrieval_query(prefix, suffix, root_dir)
    anchor   = _infer_anchor(prefix, root_dir, import_files)

    # Load all file contents
    contents: dict[str, str] = {}
    for fp in all_files:
        try:
            contents[fp] = open(fp, encoding="utf-8").read()
        except Exception:
            pass
    if not contents:
        return ""

    # BM25 scores for all files
    bm25 = bm25_scores(query, contents)

    # Score every file
    fused: dict[str, float] = {}
    for fp, content in contents.items():
        if fp in recent:
            fused[fp] = 3.0 + bm25.get(fp, 0.0) * 0.5
        elif fp in import_files:
            fused[fp] = 2.0
        else:
            if anchor and anchor != root_dir:
                dist = _path_dist(fp, anchor)
            elif import_files:
                dist = min(_path_dist(fp, ref) for ref in import_files)
            else:
                dist = fp.count(os.sep) - root_dir.count(os.sep)
            fused[fp] = bm25.get(fp, 0.0) * (1.0 / (dist + 1))

    # Top candidates for chunk scoring
    sorted_fps     = sorted(fused, key=lambda fp: fused[fp])
    candidates_top = [fp for fp in sorted_fps
                      if fp in contents][-(max_files * 3):]

    # Embed chunks
    emb_query = build_embedding_query(prefix, suffix)
    query_emb = embed_model.encode(emb_query, convert_to_tensor=True)

    raw: list[tuple[float, float, str, str]] = []
    for fp in candidates_top:
        chunks = chunk_file(contents[fp], chunk_lines)
        if not chunks:
            raw.append((0.0, fused[fp], fp, contents[fp]))
            continue
        sims = util.cos_sim(
            query_emb,
            embed_model.encode(chunks, convert_to_tensor=True,
                               show_progress_bar=False)
        )[0]
        for idx in torch.topk(sims, min(top_chunks, len(chunks))).indices.tolist():
            raw.append((float(sims[idx]), fused.get(fp, 0.0), fp, chunks[idx]))

    if not raw:
        return ""

    # Normalise chunk + file scores to same [0,1] scale then combine
    def _norm(vals):
        mn, mx = min(vals), max(vals)
        rng = mx - mn or 1.0
        return [(v - mn) / rng for v in vals]

    nc = _norm([r[0] for r in raw])
    nf = _norm([r[1] for r in raw])
    scored = sorted(
        [(0.6 * nc[i] + 0.4 * nf[i], raw[i][2], raw[i][3])
         for i in range(len(raw))],
        reverse=True,
    )

    # Greedy fill
    selected: dict[str, list[str]] = {}
    order:    list[str]            = []
    used = 0
    for _, fp, chunk in scored:
        cost = estimate_tokens(chunk)
        if used + cost > budget:
            continue
        if fp not in selected:
            if len(selected) >= max_files:
                continue
            selected[fp] = []
            order.append(fp)
        selected[fp].append(chunk)
        used += cost

    for fp in selected:
        selected[fp] = dedup_chunks(selected[fp])

    if not selected:
        return ""

    # Assemble least→most relevant
    order_sorted = sorted(order, key=lambda fp: fused.get(fp, 0.0))
    file_pairs   = []
    for fp in order_sorted:
        clean = fp[len(root_dir) + 1:]
        excerpt = "\n...\n".join(selected[fp])
        print(f"  {clean}  ({len(selected[fp])} chunk(s), score={fused.get(fp,0):.3f})")
        file_pairs.append((fp, excerpt))

    return assemble(file_pairs, root_dir, budget)


# ===========================================================================
# MAIN LOOP
# ===========================================================================

completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")
predictions_file       = os.path.join("predictions",
                                       f"{language}-{stage}-ens.jsonl")
os.makedirs("predictions", exist_ok=True)

with jsonlines.open(completion_points_file, "r") as reader:
    with jsonlines.open(predictions_file, "w") as writer:
        for datapoint in reader:
            repo_path      = datapoint["repo"].replace("/", "__")
            root_directory = os.path.join(
                "data", f"repositories-{language}-{stage}",
                f"{repo_path}-{datapoint['revision']}"
            )
            raw_prefix = datapoint["prefix"]
            raw_suffix = datapoint["suffix"]
            modified   = datapoint.get("modified", [])

            context = collect_best_context(
                root_directory, raw_prefix, raw_suffix, modified,
                args.chunk_lines, args.top_chunks,
                args.max_files, args.token_budget,
            )

            # Fallback: if context is empty, include a random file
            if not context:
                all_fp = repo_files(root_directory)
                if all_fp:
                    import random
                    fp      = random.choice(all_fp)
                    content = open(fp, encoding="utf-8").read()
                    name    = fp[len(root_directory) + 1:]
                    context = FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL,
                        file_name=name,
                        file_content=content,
                    )

            print(f"~{estimate_tokens(context)} tokens")

            writer.write({
                "context": context,
                "prefix":  build_submission_prefix(raw_prefix, root_directory),
                "suffix":  build_submission_suffix(raw_suffix),
            })