"""
Collection definition DSL: parsing, validation and blueprint generation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import define_collection
from define_collection import (
    QueryParser,
    ast_to_yaml,
    coerce_value,
    generate_names,
    infer_version,
    inspect_structural_clues,
    resolve_output_path,
    slugify,
    to_bool,
    to_number,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFINE_SCRIPT = REPO_ROOT / "scripts" / "define_collection.py"


def run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["define_collection.py", *args])
    return define_collection.main()


def test_parser_translates_a_comparison_into_criteria() -> None:
    ast = QueryParser("n_instances >= 500").parse()

    assert ast_to_yaml(ast) == [{"field": "n_instances", "op": ">=", "value": 500}]


def test_parser_translates_and_expressions() -> None:
    ast = QueryParser("task == classification AND n_features <= 100").parse()

    assert ast_to_yaml(ast) == [
        {"field": "task", "op": "==", "value": "classification"},
        {"field": "n_features", "op": "<=", "value": 100},
    ]


def test_parser_translates_or_expressions() -> None:
    ast = QueryParser("task == classification OR task == regression").parse()

    criteria = ast_to_yaml(ast)
    assert len(criteria) == 1
    branches = criteria[0]["any_of"]
    assert [branch[0]["value"] for branch in branches] == ["classification", "regression"]


def test_parser_supports_quoted_values_and_parentheses() -> None:
    ast = QueryParser("(task_subtype == 'time series regression') AND missing_rate <= 0.05").parse()

    assert ast_to_yaml(ast) == [
        {"field": "task_subtype", "op": "==", "value": "time series regression"},
        {"field": "missing_rate", "op": "<=", "value": 0.05},
    ]


def test_parser_supports_boolean_flags() -> None:
    ast = QueryParser("temporal == False").parse()

    assert ast_to_yaml(ast) == [{"field": "temporal", "op": "==", "value": False}]


@pytest.mark.parametrize(
    "expression",
    [
        "n_instances >",          # missing value
        "n_instances >= abc",     # value is not a number
        "temporal sometimes",     # invalid operator
        "n_instances >= 500 AND", # dangling operator
    ],
)
def test_parser_rejects_invalid_expressions(expression: str) -> None:
    with pytest.raises(ValueError):
        QueryParser(expression).parse()


def test_invalid_numeric_values_are_reported_with_the_field_name() -> None:
    with pytest.raises(ValueError, match="n_instances"):
        coerce_value("n_instances", "abc")


def test_invalid_boolean_values_are_reported_with_the_field_name() -> None:
    with pytest.raises(ValueError, match="temporal"):
        coerce_value("temporal", "sometimes")


def test_coerce_value_strips_surrounding_quotes() -> None:
    assert coerce_value("task", "'classification'") == "classification"
    assert coerce_value("task", '"classification"') == "classification"


def test_scalar_conversions_are_explicit_about_invalid_input() -> None:
    assert to_number("500") == 500
    assert to_number("0.05") == 0.05
    assert to_number("abc") is None
    assert to_bool("True") is True
    assert to_bool("no") is False
    assert to_bool("maybe") is None


def test_structural_clues_deduce_task_and_modifiers() -> None:
    asts = [
        QueryParser("task == classification AND task_subtype == binary").parse(),
        QueryParser("imbalance_ratio >= 5").parse(),
    ]

    task_type, task_subtype, modifiers = inspect_structural_clues(asts)

    assert task_type == "classification"
    assert task_subtype == "binary"
    assert "imbalanced" in modifiers


def test_generate_names_uses_explicit_id_and_name() -> None:
    assert generate_names("mixed", "mixed", [], "My Collection", "My Name") == ("my_collection", "My Name")


def test_generate_names_derives_readable_names_from_the_query() -> None:
    collection_id, name = generate_names("classification", "binary", ["clean"], None, None)

    assert collection_id == "binary_clean"
    assert name == "Binary Clean Classification"


def test_slugify_produces_filesystem_safe_identifiers() -> None:
    assert slugify("My Awesome Collection!") == "my_awesome_collection"
    assert slugify("   ") == "collection"


def criteria_for(task: str) -> dict:
    return {"all_of": [{"field": "task", "op": "==", "value": task}]}


def write_definition(path: Path, criteria: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"selection_criteria": criteria}, f, sort_keys=False)


def test_resolve_output_path_reuses_the_file_for_identical_criteria(tmp_path: Path) -> None:
    base_path = tmp_path / "binary_core.yaml"

    assert resolve_output_path(base_path, criteria_for("classification")) == base_path

    write_definition(base_path, criteria_for("classification"))
    assert resolve_output_path(base_path, criteria_for("classification")) == base_path


def test_resolve_output_path_never_overwrites_a_different_definition(tmp_path: Path) -> None:
    base_path = tmp_path / "binary_core.yaml"
    write_definition(base_path, criteria_for("classification"))

    second = resolve_output_path(base_path, criteria_for("regression"))
    assert second == tmp_path / "binary_core_1.yaml"

    write_definition(second, criteria_for("regression"))
    assert resolve_output_path(base_path, criteria_for("regression")) == second
    assert resolve_output_path(base_path, criteria_for("classification")) == base_path


def test_resolve_output_path_avoids_definitions_that_cannot_be_read(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    base_path = tmp_path / "binary_core.yaml"
    base_path.write_text("selection_criteria: {all_of: [\n", encoding="utf-8")

    chosen = resolve_output_path(base_path, criteria_for("classification"))

    assert chosen == tmp_path / "binary_core_1.yaml"
    assert "Could not inspect existing definition" in capsys.readouterr().out


def test_infer_version_bumps_the_minor_version(tmp_path: Path) -> None:
    path = tmp_path / "binary_core.yaml"
    path.write_text(yaml.safe_dump({"curation": {"version": "1.3"}}), encoding="utf-8")

    assert infer_version(path) == "1.4"


def test_infer_version_starts_from_the_base_version(tmp_path: Path) -> None:
    assert infer_version(tmp_path / "missing.yaml") == "1.0"


def test_infer_version_falls_back_when_the_file_cannot_be_read(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    broken = tmp_path / "binary_core.yaml"
    broken.write_text("curation: [oops\n", encoding="utf-8")

    assert infer_version(broken) == "1.0"
    assert "Could not read the version" in capsys.readouterr().out


def test_cli_requires_at_least_one_filter(project, monkeypatch, capsys: pytest.CaptureFixture) -> None:
    exit_code = run_cli(monkeypatch, "--output-dir", str(project.collections_dir))

    assert exit_code == 1
    assert "[ERROR] Please provide at least one condition filter" in capsys.readouterr().out


def test_cli_rejects_invalid_values_with_a_clear_error(project, monkeypatch, capsys) -> None:
    exit_code = run_cli(
        monkeypatch, "--filter", "n_instances >= abc", "--output-dir", str(project.collections_dir)
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "[ERROR] Invalid filter expression" in output
    assert "n_instances" in output
    assert list(project.collections_dir.glob("*.yaml")) == []


def test_cli_writes_a_complete_blueprint(project, monkeypatch) -> None:
    exit_code = run_cli(
        monkeypatch,
        "--filter",
        "task == classification AND task_subtype == binary",
        "--id",
        "binary_core",
        "--name",
        "Binary Core",
        "--output-dir",
        str(project.collections_dir),
    )

    assert exit_code == 0
    blueprint_path = project.collections_dir / "binary_core.yaml"
    blueprint = yaml.safe_load(blueprint_path.read_text(encoding="utf-8"))
    assert blueprint["collection_id"] == "binary_core"
    assert blueprint["name"] == "Binary Core"
    assert blueprint["selection_expression"] == "task == classification AND task_subtype == binary"
    fields = {condition["field"] for condition in blueprint["selection_criteria"]["all_of"]}
    assert {"task", "task_subtype"} <= fields
    assert blueprint["output_fields"][0] == "dataset_id"
    assert {"description", "preferred_sorting", "output_fields", "curation"} <= set(blueprint)
    assert blueprint["curation"]["version"] == "1.0"


def test_cli_keeps_one_file_for_identical_definitions(project, monkeypatch) -> None:
    args = ("--filter", "task == regression", "--id", "reg_core", "--output-dir", str(project.collections_dir))

    assert run_cli(monkeypatch, *args) == 0
    assert run_cli(monkeypatch, *args) == 0
    assert sorted(entry.name for entry in project.collections_dir.glob("*.yaml")) == ["reg_core.yaml"]


def test_cli_writes_a_new_file_when_the_criteria_change(project, monkeypatch) -> None:
    first = run_cli(
        monkeypatch,
        "--filter",
        "task == regression",
        "--id",
        "reg_core",
        "--output-dir",
        str(project.collections_dir),
    )
    second = run_cli(
        monkeypatch,
        "--filter",
        "task == regression AND n_instances <= 2000",
        "--id",
        "reg_core",
        "--output-dir",
        str(project.collections_dir),
    )

    assert (first, second) == (0, 0)
    assert sorted(entry.name for entry in project.collections_dir.glob("*.yaml")) == [
        "reg_core.yaml",
        "reg_core_1.yaml",
    ]


def test_cli_process_exits_with_a_status_code_and_no_traceback(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(DEFINE_SCRIPT), "--filter", "n_instances >= abc", "--output-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 1
    assert "[ERROR] Invalid filter expression" in result.stdout
    assert "Traceback" not in result.stderr
