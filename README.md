# Token Count

## execute

```sh
$ python token_count.py path/to/watch --watch
```

## for development

Download tokenizer

```sh
$ cd tokenizer
$ HF_ACCESS_TOKEN="your token"
$ curl -L -H "Authorization: Bearer $HF_ACCESS_TOKEN" https://huggingface.co/pfnet/plamo-2.1-8b-cpt/resolve/main/tokenizer.jsonl -o tokenizer.jsonl
$ curl -L -H "Authorization: Bearer $HF_ACCESS_TOKEN" https://huggingface.co/pfnet/plamo-2.1-8b-cpt/resolve/main/tokenization_plamo.py -o tokenization_plamo.py
```

Type check

```sh
$ uv run mypy .
```

Format check

```sh
$ ruff check . --fix
```
