from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helpers.hotwords import load_hotwords
from helpers.initial_prompt import load_initial_prompt
from helpers.spelling_map import apply_spelling_rules, load_spelling_map


def test_load_hotwords_normalization(tmp_path):
    hotwords_file = tmp_path / "hotwords.txt"
    hotwords_file.write_text("dragon, wizard\n rogue , bard \n\n", encoding="utf-8")

    result = load_hotwords(str(hotwords_file))

    assert result == "dragon, wizard, rogue, bard"


def test_load_hotwords_missing_or_empty(tmp_path):
    missing_path = tmp_path / "missing.txt"

    assert load_hotwords(str(missing_path)) is None
    assert load_hotwords(None) is None

    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("\n\n", encoding="utf-8")

    assert load_hotwords(str(empty_file)) is None


def test_load_initial_prompt_trimming(tmp_path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("   Seek the orb.\n\n", encoding="utf-8")

    result = load_initial_prompt(str(prompt_file))

    assert result == "Seek the orb."


def test_load_initial_prompt_missing_or_blank(tmp_path):
    missing_path = tmp_path / "missing_prompt.txt"

    assert load_initial_prompt(str(missing_path)) is None
    assert load_initial_prompt(None) is None

    blank_file = tmp_path / "blank_prompt.txt"
    blank_file.write_text("   \n\n", encoding="utf-8")

    assert load_initial_prompt(str(blank_file)) is None


def test_spelling_map_and_apply_rules(tmp_path):
    csv_content = """wrong,right\nGoblin,goblin\nELF,Elf King\n ,  \n"""
    csv_file = tmp_path / "spelling.csv"
    csv_file.write_text(csv_content, encoding="utf-8")

    rules = load_spelling_map(str(csv_file))

    assert rules == [("Goblin", "goblin"), ("ELF", "Elf King")]

    sample_text = (
        "The goblin saw a Goblin near the ELF. "
        "Another elf lurked on a shelf."
    )

    updated_text = apply_spelling_rules(sample_text, rules)

    assert (
        updated_text
        == "The goblin saw a goblin near the Elf King. "
        "Another Elf King lurked on a shelf."
    )
