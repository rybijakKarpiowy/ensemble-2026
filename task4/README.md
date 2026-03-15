# task4 — Submission generation

This folder contains the script used to generate the submission file (`.npz`) for the test split.

## Requirements

- Python 3
- Any project dependencies required by `make_submission.py`

## Usage

### 1) Generate submission with debug artifacts

Runs inference on the `test` input, writes the submission to `outputs/submission.npz`, and saves debug outputs to `outputs/debug`.

```bash
python make_submission.py --input test --submission outputs/submission.npz --out-dir outputs/debug --save-debug
```

### 2) Generate submission (NPZ only)

Runs inference on the `test` input and writes only the submission NPZ file.

```bash
python make_submission.py --input test --submission submission.npz
```

## Output

- `submission.npz` (or `outputs/submission.npz`): the file to submit.
- `outputs/debug/` (only when using `--out-dir ... --save-debug`): debug artifacts to help inspect intermediate results.
