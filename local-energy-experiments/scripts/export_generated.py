"""Export the saved generated_code fields from experiment JSON to Python files."""

import argparse
import json
import re
from pathlib import Path


def name(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A result JSON or directory of result JSON files")
    parser.add_argument("--output", type=Path, required=True, help="A new or empty directory")
    args = parser.parse_args()
    sources = sorted(args.input.rglob("*.json")) if args.input.is_dir() else [args.input]
    if not sources:
        parser.error("No result JSON files found.")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output must be a new or empty directory.")

    programs = {}
    skipped = 0
    for source in sources:
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            parser.error(f"Expected a local experiment dataset: {source.name}")
        for row in data["results"]:
            code = row.get("generated_code")
            if not code:
                skipped += 1
                continue
            relative = Path(name(row["model"]), name(row["task"]), name(row["strategy"]),
                            f"run_{name(row['run'])}.py")
            if relative in programs:
                parser.error(f"Duplicate trial identity: {relative}")
            programs[relative] = code

    for relative, code in programs.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
    print(f"Exported {len(programs)} saved programs; {skipped} trials had no extracted code.")


if __name__ == "__main__":
    main()
