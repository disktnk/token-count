import argparse
import io
import os
import time
from pathlib import Path
from typing import Optional

import frontmatter  # type: ignore
from ruamel.yaml import YAML
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
from tokenizer.tokenization_plamo import Plamo2Tokenizer

tokenizer: Optional[Plamo2Tokenizer] = None


def initialize_tokenizer() -> Plamo2Tokenizer:
    global tokenizer
    if tokenizer is None:
        tokenizer = Plamo2Tokenizer(vocab_file="tokenizer/tokenizer.jsonl")
    return tokenizer


def count_tokens(text: str) -> int:
    if tokenizer is None:
        raise ValueError("Tokenizer is not initialized.")
    encoded = tokenizer.encode(text)
    return len(encoded)


def _order_metadata(metadata: dict) -> dict:
    """Order metadata keys."""
    order = ["date", "title"]
    # first "date", "title", then others in alphabetical order
    ordered_metadata = {key: metadata[key] for key in order if key in metadata}
    other_keys = sorted([key for key in metadata if key not in order])
    sorted_metadata = {key: metadata[key] for key in other_keys}
    return {**ordered_metadata, **sorted_metadata}


def insert_token_count(path: Path) -> None:
    with open(path, "r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    if "title" not in post:
        return
    if "date" not in post:
        return

    num_tokens = count_tokens(post.content)
    prev_tokens = post.get("tokens", -1)

    if prev_tokens == num_tokens:
        return

    # Update tokens in metadata
    post["tokens"] = num_tokens

    # Order metadata and ensure string types for date/title
    post.metadata = _order_metadata(post.metadata)
    for k in ["date", "title"]:
        if k in post.metadata:
            post.metadata[k] = str(post.metadata[k])

    # Use ruamel.yaml to maintain formatting and order
    yaml = YAML()
    yaml.default_flow_style = False
    temp_path = path.with_suffix(".tmp.md")
    try:
        meta_stream = io.StringIO()
        yaml.dump(post.metadata, meta_stream)
        yaml_text = meta_stream.getvalue().strip()
        new_content = f"---\n{yaml_text}\n---\n{post.content}\n"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        temp_path.rename(path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"Error processing {path}: {e}")


def process_markdowns(markdown_paths: list[Path]) -> None:
    for path in markdown_paths:
        if not path.is_file():
            continue
        insert_token_count(path)


def update(target: Path) -> None:
    if target.is_dir():
        markdown_paths = list(target.rglob("*.md"))
        process_markdowns(markdown_paths)

    elif target.is_file():
        if not target.suffix == ".md":
            print(f"Error: {target} is not a markdown file.")
        insert_token_count(target)


class MarkdownHandler(FileSystemEventHandler):

    updated_table: dict[Path, float] = {}

    def on_modified(self, event: FileSystemEvent) -> None:
        modified_path = Path(str(event.src_path))
        if modified_path.suffix != ".md":
            return
        if modified_path.name.endswith(".tmp.md"):
            return

        modified_time = modified_path.stat().st_mtime
        previous_modified_time = self.updated_table.setdefault(modified_path, 0.0)
        if previous_modified_time + 1 > modified_time:
            return
        self.updated_table[modified_path] = modified_time

        if modified_path.stat().st_mtime < time.time() - 1:
            return

        print(f"File modified: {modified_path}")
        update(modified_path)


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl markdown files in a directory.")
    parser.add_argument(
        "target",
        type=str,
        help="The target file or directory to crawl for markdown files.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch the target directory for changes and update token counts automatically.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arg()
    target = Path(args.target)

    initialize_tokenizer()

    if args.watch:
        observer = Observer()
        observer.schedule(MarkdownHandler(), str(target), recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    update(target)
