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
argparser.add_argument("--strategy",     type=str, default="random",   help="Context collection strategy")
argparser.add_argument("--trim-prefix",  action="store_true",          help="Trim the prefix to 10 lines")
argparser.add_argument("--trim-suffix",  action="store_true",          help="Trim the suffix to 10 lines")
argparser.add_argument("--prefix-mode",  type=str, default="smart",    choices=["smart", "simple", "full"])
argparser.add_argument("--suffix-mode",  type=str, default="full",     choices=["full", "filtered", "none"])
# smart_multi params
argparser.add_argument("--chunk-lines",  type=int, default=30,   help="Lines per chunk when scoring file sections")
argparser.add_argument("--top-chunks",   type=int, default=3,    help="Top-K chunks to keep per file")
argparser.add_argument("--max-files",    type=int, default=4,    help="Max files to include in context")
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

# Tokens used to separate files in the context
FILE_SEP_SYMBOL    = "<|file_sep|>"
FILE_COMPOSE_FORMAT = "{file_sep}{file_name}\n{file_content}"

# Embedding model — loaded once at startup; only needed for embedding-based strategies
embed_model = (
    SentenceTransformer('all-MiniLM-L6-v2')
    if strategy in ("embeddings", "smart_multi")
    else None
)


# ===========================================================================
# PREFIX / SUFFIX PROCESSING
# ===========================================================================

def extract_local_imports(prefix: str, root_dir: str) -> list[str]:
    """
    Parse import statements in `prefix` and resolve them to actual file paths
    inside `root_dir`.  Only returns paths that exist on disk.

    e.g.  `from utils.auth import Foo`  →  looks for  utils/auth.py
    """
    resolved = []
    try:
        tree = ast.parse(prefix)
    except SyntaxError:
        return resolved

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            rel_path  = node.module.replace(".", os.sep) + extension
            candidate = os.path.join(root_dir, rel_path)
            if os.path.isfile(candidate):
                resolved.append(candidate)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                rel_path  = alias.name.replace(".", os.sep) + extension
                candidate = os.path.join(root_dir, rel_path)
                if os.path.isfile(candidate):
                    resolved.append(candidate)

    return list(set(resolved))


def get_prefix_for_submission(prefix: str, mode: str, limit: int = 10) -> str:
    """
    Process the prefix that is written to the submission file.

    - 'full'   → unchanged
    - 'simple' → last `limit` lines
    - 'smart'  → import lines + last `limit` lines (deduped, order preserved)
    """
    lines = prefix.splitlines()
    if mode == "full" or len(lines) <= limit:
        return prefix
    if mode == "simple":
        return "\n".join(lines[-limit:])
    # smart
    import_lines = [l for l in lines if l.startswith(("import ", "from "))]
    tail_lines   = lines[-limit:]
    combined     = list(dict.fromkeys(import_lines + tail_lines))
    return "\n".join(combined)


def get_retrieval_prefix(prefix: str, limit: int = 20) -> str:
    """
    Build the query prefix used for file-level retrieval:
      - ALL import lines from the full prefix  (for import-aware lookup)
      - PLUS the last `limit` non-import lines (the immediate code context)
    """
    lines        = prefix.splitlines()
    import_lines = [l for l in lines if l.startswith(("import ", "from "))]
    tail_lines   = lines[-limit:] if len(lines) > limit else lines
    combined     = list(dict.fromkeys(import_lines + tail_lines))
    return "\n".join(combined)


def get_retrieval_suffix(suffix: str, mode: str) -> str:
    """
    Return the suffix portion to use for retrieval queries.

    - 'none'     → always empty
    - 'filtered' → empty when suffix is only closing brackets / whitespace
    - 'full'     → unchanged
    """
    if mode == "none":
        return ""
    if mode == "filtered":
        if re.match(r"^[\s}\)\]:,;]*$", suffix.strip()):
            return ""
    return suffix


def get_embedding_query(prefix: str, suffix: str) -> str:
    """
    Build the embedding query used for *chunk* scoring inside files:
      - prefix WITHOUT import lines  (pure logic near the cursor)
      - plus the filtered suffix
    This keeps the query focused on what the code is doing, not its imports.
    """
    lines      = prefix.splitlines()
    code_lines = [l for l in lines if not l.startswith(("import ", "from "))]
    tail       = "\n".join(code_lines[-20:]) if len(code_lines) > 20 else "\n".join(code_lines)
    clean_suf  = get_retrieval_suffix(suffix, "filtered")
    return (tail + "\n" + clean_suf).strip()


def trim_suffix(suffix: str) -> str:
    """Keep only the first 10 lines of the suffix (legacy helper)."""
    lines = suffix.split("\n")
    return "\n".join(lines[:10]) if len(lines) > 10 else suffix


# ===========================================================================
# FILE COLLECTION HELPERS
# ===========================================================================

def all_repo_files(root_dir: str, min_lines: int = 10) -> list[str]:
    """Return all language files under `root_dir` that have at least `min_lines`."""
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
    """
    Return existing, long-enough files from the `modified` list.
    List is expected most-recent-first; order is preserved.
    """
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


# ===========================================================================
# CHUNK SCORING  (embedding-based)
# ===========================================================================

def chunk_file(content: str, chunk_size: int = 30) -> list[str]:
    """Split `content` into overlapping windows of `chunk_size` lines (50 % overlap)."""
    lines  = content.splitlines()
    chunks = []
    step   = max(1, chunk_size // 2)
    for i in range(0, len(lines), step):
        chunk = "\n".join(lines[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


# ===========================================================================
# LEGACY SINGLE-FILE STRATEGIES
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
            content = open(fp, encoding="utf-8").read()
            corpus.append(tokenize(content))
            file_names.append(fp)
        except Exception:
            pass

    if not file_names:
        return None

    query  = tokenize(prefix + " " + suffix)
    scores = BM25Okapi(corpus).get_scores(query)
    return file_names[scores.argmax()]


def find_embedding_file(root_dir: str, query_text: str, min_lines: int = 10) -> str:
    file_names, corpus_contents = [], []
    for fp in all_repo_files(root_dir, min_lines):
        try:
            content = open(fp, encoding="utf-8").read()
            corpus_contents.append(content)
            file_names.append(fp)
        except Exception:
            pass

    if not file_names:
        return None

    query_emb   = embed_model.encode(query_text, convert_to_tensor=True)
    corpus_embs = embed_model.encode(corpus_contents, convert_to_tensor=True)
    scores      = util.cos_sim(query_emb, corpus_embs)[0]
    return file_names[torch.argmax(scores).item()]


# ===========================================================================
# SMART MULTI-FILE STRATEGY
# ===========================================================================

def collect_smart_context(
    root_dir:         str,
    prefix:           str,
    suffix:           str,
    recent_filenames: list[str],
    chunk_size:       int = 30,
    top_chunks:       int = 3,
    max_files:        int = 4,
    token_budget:     int = 6000,
) -> str:
    """
    Build a rich multi-file context in four steps:

    Step 1 — Candidate files
        a) Files resolved directly from import statements in the prefix
        b) Recently modified files (ordered most-recent-first)
        Deduplicated, import-resolved files take priority.
        Falls back to all repo files when there are too few candidates.

    Step 2 — Embedding query
        prefix (WITHOUT import lines) + filtered suffix.
        Keeps the query focused on what the code is *doing*, not its deps.

    Step 3 — Chunk scoring
        Each candidate file is split into overlapping 30-line windows.
        All chunks across all files are scored against the embedding query.

    Step 4 — Greedy budget fill
        Best-scoring chunks are selected until the token budget is exhausted.
        Results are grouped by file and formatted with <|file_sep|> headers.
    """

    # ------------------------------------------------------------------
    # Step 1 — Candidate file list
    # ------------------------------------------------------------------
    import_files = extract_local_imports(prefix, root_dir)   # high-precision
    recent_files = collect_recent_files(root_dir, recent_filenames)

    seen: set[str] = set()
    candidates: list[str] = []
    for fp in import_files + recent_files:
        if fp not in seen:
            seen.add(fp)
            candidates.append(fp)

    # Pad with full repo scan if too few candidates
    if len(candidates) < 2:
        for fp in all_repo_files(root_dir):
            if fp not in seen:
                seen.add(fp)
                candidates.append(fp)

    if not candidates:
        return ""

    # ------------------------------------------------------------------
    # Step 2 — Build embedding query (code intent, no imports)
    # ------------------------------------------------------------------
    emb_query = get_embedding_query(prefix, suffix)
    query_emb = embed_model.encode(emb_query, convert_to_tensor=True)

    # ------------------------------------------------------------------
    # Step 3 — Score chunks from each candidate
    # ------------------------------------------------------------------
    # (score, file_path, chunk_text)
    scored_chunks: list[tuple[float, str, str]] = []

    scan_limit = max_files * 4   # scan more files than we'll keep, then filter
    for fp in candidates[:scan_limit]:
        try:
            content = open(fp, encoding="utf-8").read()
        except Exception:
            continue

        chunks = chunk_file(content, chunk_size)
        if not chunks:
            continue

        chunk_embs = embed_model.encode(chunks, convert_to_tensor=True)
        sims       = util.cos_sim(query_emb, chunk_embs)[0]

        k = min(top_chunks, len(chunks))
        for idx in torch.topk(sims, k).indices.tolist():
            scored_chunks.append((sims[idx].item(), fp, chunks[idx]))

    # Global rank: best chunks first
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # Step 4 — Greedy token-budget fill, grouped by file
    # ------------------------------------------------------------------
    # Keep insertion order so import-resolved files stay near the top
    selected: dict[str, list[str]] = {}
    files_used  = 0
    tokens_used = 0

    for score, fp, chunk in scored_chunks:
        cost = estimate_tokens(chunk)
        if tokens_used + cost > token_budget:
            continue
        if fp not in selected:
            if files_used >= max_files:
                continue
            selected[fp] = []
            files_used  += 1
        selected[fp].append(chunk)
        tokens_used += cost

    # ------------------------------------------------------------------
    # Assemble final context string
    # ------------------------------------------------------------------
    parts = []
    for fp, chunks in selected.items():
        clean_name   = fp[len(root_dir) + 1:]
        file_excerpt = "\n...\n".join(chunks)
        parts.append(
            FILE_COMPOSE_FORMAT.format(
                file_sep=FILE_SEP_SYMBOL,
                file_name=clean_name,
                file_content=file_excerpt,
            )
        )
        print(f"  [smart_multi] included: {clean_name}  ({len(chunks)} chunk(s))")

    return "".join(parts)


# ===========================================================================
# MAIN LOOP
# ===========================================================================

completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")

prediction_file_name = f"{language}-{stage}-{strategy}-p_{args.prefix_mode}-s_{args.suffix_mode}"
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

            # ----------------------------------------------------------
            # Dispatch to selected strategy
            # ----------------------------------------------------------

            if strategy == "random":
                file_name    = find_random_file(root_directory)
                file_content = open(file_name, encoding="utf-8").read()
                clean_name   = file_name[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=clean_name, file_content=file_content
                )
                print(f"Picked file: {clean_name}")

            elif strategy == "bm25":
                search_prefix = get_retrieval_prefix(raw_prefix)
                search_suffix = get_retrieval_suffix(raw_suffix, args.suffix_mode)
                file_name     = find_bm25_file(root_directory, search_prefix, search_suffix)
                file_content  = open(file_name, encoding="utf-8").read()
                clean_name    = file_name[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=clean_name, file_content=file_content
                )
                print(f"Picked file: {clean_name}")

            elif strategy == "recent":
                recent_filenames = datapoint.get('modified', [])
                recent           = collect_recent_files(root_directory, recent_filenames)
                file_name        = recent[0] if recent else find_random_file(root_directory)
                file_content     = open(file_name, encoding="utf-8").read()
                clean_name       = file_name[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=clean_name, file_content=file_content
                )
                print(f"Picked file: {clean_name}")

            elif strategy == "embeddings":
                search_prefix = get_retrieval_prefix(raw_prefix)
                search_suffix = get_retrieval_suffix(raw_suffix, args.suffix_mode)
                file_name     = find_embedding_file(
                    root_directory, search_prefix + "\n" + search_suffix
                )
                file_content  = open(file_name, encoding="utf-8").read()
                clean_name    = file_name[len(root_directory) + 1:]
                context = FILE_COMPOSE_FORMAT.format(
                    file_sep=FILE_SEP_SYMBOL, file_name=clean_name, file_content=file_content
                )
                print(f"Picked file: {clean_name}")

            elif strategy == "smart_multi":
                context = collect_smart_context(
                    root_dir          = root_directory,
                    prefix            = raw_prefix,
                    suffix            = raw_suffix,
                    recent_filenames  = datapoint.get('modified', []),
                    chunk_size        = args.chunk_lines,
                    top_chunks        = args.top_chunks,
                    max_files         = args.max_files,
                    token_budget      = args.token_budget,
                )
                # Ultimate fallback
                if not context:
                    file_name    = find_random_file(root_directory)
                    file_content = open(file_name, encoding="utf-8").read()
                    clean_name   = file_name[len(root_directory) + 1:]
                    context = FILE_COMPOSE_FORMAT.format(
                        file_sep=FILE_SEP_SYMBOL, file_name=clean_name, file_content=file_content
                    )
                print(f"[smart_multi] total context: ~{estimate_tokens(context)} tokens")

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # ----------------------------------------------------------
            # Build submission dict
            # ----------------------------------------------------------
            submission = {"context": context}

            if args.trim_prefix:
                submission["prefix"] = get_prefix_for_submission(raw_prefix, "smart")
            if args.trim_suffix:
                submission["suffix"] = trim_suffix(raw_suffix)

            writer.write(submission)