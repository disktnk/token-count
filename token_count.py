import argparse
import re
from pathlib import Path

import frontmatter

from tokenizer.tokenization_plamo import Plamo2Tokenizer

tokenizer: Plamo2Tokenizer = None


def initialize_tokenizer() -> Plamo2Tokenizer:
    global tokenizer
    if tokenizer is None:
        tokenizer = Plamo2Tokenizer(vocab_file="tokenizer/tokenizer.jsonl")
    return tokenizer


def count_tokens(text: str) -> int:
    encoded = tokenizer.encode(text)
    return len(encoded)


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

    # Memo:
    # python-frontmatter is omit "'" in dumping, and unexpectedly changes date
    # format, cannot handler these customizations. So dump string directly.
    token_str = f"tokens: {num_tokens}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = "(---\n.*?)(---\n)"
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        print(f"Error: No frontmatter found in {path}.")
        return
    
    if prev_tokens == -1:
       content = re.sub(pattern, f"\\1{token_str}\n---\n", content, count=1, flags=re.DOTALL)
    else:
        prev_token_str = f"tokens: {prev_tokens}"
        content = content.replace(prev_token_str, token_str)

    temp_path = path.with_suffix(".tmp.md")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(content)
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


def crawl_markdown_files(target_dir: str) -> list[Path]:
    dir = Path(target_dir)
    return list(dir.rglob("*.md"))


def update(target: Path) -> None:
    if target.is_dir():
        markdown_paths = crawl_markdown_files(target)
        process_markdowns(markdown_paths)

    elif target.is_file():
        if not target.suffix == ".md":
            print(f"Error: {target} is not a markdown file.")
        insert_token_count(target)


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl markdown files in a directory.")
    parser.add_argument(
        "target",
        type=str,
        help="The target file or directory to crawl for markdown files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arg()
    target = Path(args.target)

    initialize_tokenizer()

    update(target)
