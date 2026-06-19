#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import subprocess

import yaml

# Directory paths matching project architecture
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTIONS_DIR = ROOT / "collections"

NUMERIC_FIELDS = {
    "n_instances",
    "n_features",
    "n_classes",
    "missing_rate",
    "imbalance_ratio",
    "p_to_n_ratio",
}

BOOL_FIELDS = {
    "temporal",
    "grouped",
    "academic_usage",
    "iid_assumed",
}

PRESET_OUTPUT_FIELDS = [
    "dataset_id",
    "name",
    "task",
    "task_subtype",
    "target_variable",
    "source_repository",
    "source_id",
    "n_instances",
    "n_features",
    "n_classes",
    "missing_rate",
    "imbalance_ratio",
    "temporal",
    "grouped",
    "academic_usage",
    "license",
    "status",
]

TOKEN_RE = re.compile(
    r"""
    \s*
    (
        \(|\)|
        <=|>=|==|!=|<|>|
        \bAND\b|\bOR\b|
        "(?:[^"\\]|\\.)*"|
        '(?:[^'\\]|\\.)*'|
        [^\s()<>!=]+
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ----------------------------
# Format Helpers
# ----------------------------
def normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def to_number(value: str) -> float | int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text) if "." in text or "e" in text.lower() else int(text)
    except ValueError:
        return None


def to_bool(value: str) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def slugify(text: str) -> str:
    out = []
    prev_dash = False
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        else:
            if not prev_dash:
                out.append("_")
                prev_dash = True
    slug = "".join(out).strip("_")
    return slug or "collection"


def coerce_value(field: str, raw_val: str) -> Any:
    field = field.strip().lower()
    if field in NUMERIC_FIELDS:
        return to_number(raw_val)
    if field in BOOL_FIELDS:
        return to_bool(raw_val)
    text = str(raw_val).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1]
    return text


def tokenize(expr: str) -> list[str]:
    return [t.strip() for t in TOKEN_RE.findall(expr) if t.strip()]


def infer_version(collection_path: Path, base_version: str = "1.0") -> str:
    if not collection_path.exists():
        return base_version
    try:
        data = yaml.safe_load(collection_path.read_text(encoding="utf-8"))
        existing = str(data.get("curation", {}).get("version", base_version))
        parts = existing.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            major, minor = int(parts[0]), int(parts[1])
            return f"{major}.{minor + 1}"
    except Exception:
        pass
    return base_version


def resolve_output_path(base_path: Path, new_criteria: dict) -> Path:
    """Handles identical name configurations dynamically without immediate overwrites."""
    if not base_path.exists():
        return base_path

    try:
        with base_path.open("r", encoding="utf-8") as f:
            existing = yaml.safe_load(f)
        if existing.get("selection_criteria") == new_criteria:
            return base_path  # Identical design parameters, overwrite safe
    except Exception:
        pass

    parent = base_path.parent
    stem = base_path.stem
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}.yaml"
        if not candidate.exists():
            return candidate
        try:
            with candidate.open("r", encoding="utf-8") as f:
                existing_v = yaml.safe_load(f)
            if existing_v.get("selection_criteria") == new_criteria:
                return candidate
        except Exception:
            pass
        counter += 1


# ----------------------------
# Grammar AST Parser
# ----------------------------
class QueryParser:
    def __init__(self, expr: str):
        self.expr = expr
        self.tokens = tokenize(expr)
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError(f"Unexpected end of expression: {self.expr}")
        self.pos += 1
        return tok

    def match(self, expected: str) -> bool:
        tok = self.peek()
        if tok is not None and tok.upper() == expected.upper():
            self.pos += 1
            return True
        return False

    def parse(self) -> dict[str, Any]:
        node = self.parse_or()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token '{self.peek()}' in: {self.expr}")
        return node

    def parse_or(self) -> dict[str, Any]:
        items = [self.parse_and()]
        while self.match("OR"):
            items.append(self.parse_and())
        if len(items) == 1:
            return items[0]
        return {"type": "or", "items": items}

    def parse_and(self) -> dict[str, Any]:
        items = [self.parse_factor()]
        while self.match("AND"):
            items.append(self.parse_factor())
        if len(items) == 1:
            return items[0]
        return {"type": "and", "items": items}

    def parse_factor(self) -> dict[str, Any]:
        if self.match("("):
            node = self.parse_or()
            if not self.match(")"):
                raise ValueError(f"Missing ')' in: {self.expr}")
            return node
        return self.parse_comparison()

    def parse_comparison(self) -> dict[str, Any]:
        field = self.take().strip().lower()
        op = self.take()
        value_token = self.take()

        if op not in {"<=", ">=", "==", "!=", "<", ">"}:
            raise ValueError(f"Invalid operator '{op}' in: {self.expr}")

        value = coerce_value(field, value_token)
        return {"type": "cmp", "field": field, "op": op, "value": value}


def ast_to_yaml(node: dict[str, Any]) -> list[dict[str, Any]]:
    t = node["type"]
    if t == "cmp":
        return [{"field": node["field"], "op": node["op"], "value": node["value"]}]
    if t == "and":
        out = []
        for x in node["items"]:
            out.extend(ast_to_yaml(x))
        return out
    if t == "or":
        # Safe structural fallback for compound logic blocks
        return [{"any_of": [ast_to_yaml(x) for x in node["items"]]}]
    return []


def collect_cmp_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    if node["type"] == "cmp":
        return [node]
    out: list[dict[str, Any]] = []
    if "items" in node:
        for child in node["items"]:
            out.extend(collect_cmp_nodes(child))
    return out


# ----------------------------
# Context Clue Extraction Engine
# ----------------------------
def inspect_structural_clues(ast_nodes: list[dict[str, Any]]) -> tuple[str, str, list[str]]:
    """Analyzes expression trees directly to deduce context requirements."""
    task_type = "mixed"
    task_subtype = "mixed"
    modifiers: list[str] = []

    for node in ast_nodes:
        comps = collect_cmp_nodes(node)
        for c in comps:
            field = c["field"]
            op = c["op"]
            value = c["value"]

            if field == "task" and op == "==":
                task_type = str(value).lower()
            elif field == "task_subtype" and op == "==":
                task_subtype = str(value).lower()
            elif field == "imbalance_ratio" and op in {">", ">="} and isinstance(value, (int, float)) and value >= 3:
                modifiers.append("imbalanced")
            elif field == "missing_rate" and op in {"<", "<="} and isinstance(value, (int, float)) and value <= 0.05:
                modifiers.append("clean")
            elif field == "n_instances" and op in {"<", "<="} and isinstance(value, int) and value <= 1500:
                modifiers.append("small")
            elif field == "n_instances" and op in {">", ">="} and isinstance(value, int) and value >= 5000:
                modifiers.append("large")
            elif field == "n_features" and op in {">", ">="} and isinstance(value, int) and value >= 50:
                modifiers.append("high_dimensional")
            elif field == "n_features" and op in {"<", "<="} and isinstance(value, int) and value <= 10:
                modifiers.append("low_dimensional")
            elif field == "temporal" and value is True:
                modifiers.append("temporal")
            elif field == "grouped" and value is True:
                modifiers.append("grouped")

    # Lock down implicit classification contexts if subtype strings yield early indicators
    if task_subtype in {"binary", "multiclass"} and task_type == "mixed":
        task_type = "classification"

    seen = set()
    unique_modifiers = [m for m in modifiers if not (m in seen or seen.add(m))]
    return task_type, task_subtype, unique_modifiers


def generate_names(
    task_type: str, task_subtype: str, modifiers: list[str], explicit_id: str | None, explicit_name: str | None
) -> tuple[str, str]:
    
    if explicit_id:
        collection_id = slugify(explicit_id)
    else:
        base = "mixed" if task_type == "mixed" else (task_subtype if task_subtype != "mixed" else task_type)
        collection_id = slugify("_".join([base] + modifiers))

    if explicit_name:
        name = explicit_name.strip()
    else:
        if task_type == "mixed":
            lead, suffix = "Mixed", "Benchmark"
        elif task_type == "classification":
            lead = "Binary" if task_subtype == "binary" else ("Multiclass" if task_subtype == "multiclass" else "Classification")
            suffix = "Classification"
        else:
            lead = "Time Series" if task_subtype == "time_series_regression" else "Regression"
            suffix = "Regression"

        pretty_mods = [m.replace("_", " ").title() for m in modifiers]
        words = [lead] + pretty_mods
        if lead != suffix:
            words.append(suffix)
        name = " ".join(words)

    return collection_id, name

def get_git_author() -> str:
    """
    Returns the configured Git username.
    Falls back to 'user' if Git is unavailable or not configured.
    """
    try:
        result = subprocess.run(
            ["git", "config", "--get", "user.name"],
            capture_output=True,
            text=True,
            timeout=3
        )

        author = result.stdout.strip()

        if author:
            return author

    except Exception:
        pass

    return "user"

# ----------------------------
# Execution Entry Point
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Define requirements specification for targeted dataset collections.")
    parser.add_argument("--filter", action="append", default=[], help='Criteria expression, e.g. "imbalance_ratio >= 5"')
    parser.add_argument("--id", default=None, help="Explicit collection ID target.")
    parser.add_argument("--name", default=None, help="Explicit human-readable collection name.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COLLECTIONS_DIR)
    args = parser.parse_args()

    if not args.filter:
        print("❌ Please provide at least one condition filter expression using --filter")
        return 1

    ast_nodes = []
    raw_expressions = []
    selection_criteria_list = []

    for f_expr in args.filter:
        query_parser = QueryParser(f_expr)
        ast = query_parser.parse()
        ast_nodes.append(ast)
        raw_expressions.append(f_expr.strip())
        selection_criteria_list.extend(ast_to_yaml(ast))

    # Infer properties completely from rule inputs
    task_type, task_subtype, modifiers = inspect_structural_clues(ast_nodes)
    
    # Prepend basic environment rules if not declared explicitly in your filters
    context_filters = []
    existing_fields = {c["field"] for n in ast_nodes for c in collect_cmp_nodes(n)}
    if task_type in {"classification", "regression"} and "task" not in existing_fields:
        context_filters.append({"field": "task", "op": "==", "value": task_type})
    if task_subtype and task_subtype != "mixed" and "task_subtype" not in existing_fields:
        context_filters.append({"field": "task_subtype", "op": "==", "value": task_subtype})

    final_criteria = {"all_of": context_filters + selection_criteria_list}
    selection_expression = " AND ".join(raw_expressions)

    collection_id, collection_name = generate_names(task_type, task_subtype, modifiers, args.id, args.name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.output_dir / f"{collection_id}.yaml"
    
    version = infer_version(base_path)

    # Automatically choose strategic target orderings based on filter focus points
    sorting = []
    if "imbalance_ratio" in existing_fields:
        sorting.append({"field": "imbalance_ratio", "order": "desc"})
    if "n_instances" in existing_fields:
        sorting.append({"field": "n_instances", "order": "asc"})
    else:
        sorting.append({"field": "n_instances", "order": "desc"})

    yaml_data = {
        "collection_id": collection_id,
        "name": collection_name,
        "description": f"Automatically generated collection of {collection_name.lower()} datasets from configuration criteria.",
        "selection_expression": selection_expression,
        "selection_criteria": final_criteria,
        "preferred_sorting": sorting,
        "output_fields": PRESET_OUTPUT_FIELDS,
        "curation": {
            "version": version,
            "author": get_git_author(),
            "date": datetime.now().strftime("%Y-%m-%d"),
        },
    }

    # Resolve filename modifications and sync contents
    output_path = resolve_output_path(base_path, final_criteria)
    final_id = output_path.stem
    yaml_data["collection_id"] = final_id

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, indent=2, default_flow_style=False)

    print(f"✅ Specification manifest saved to: {output_path}")
    print(f"   Collection ID   : {final_id}")
    print(f"   Collection Name : {collection_name}")


if __name__ == "__main__":
    main()