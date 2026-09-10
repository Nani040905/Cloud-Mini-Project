"""
scripts/generate_readme_tables.py

Reads results/final_results.csv and generates formatted GitHub Markdown tables
for README.md and documentation to ensure 100% data synchronization.
"""

import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_CSV = os.path.join(PROJECT_ROOT, "results", "final_results.csv")
README_PATH = os.path.join(PROJECT_ROOT, "README.md")


def format_markdown_table(df: pd.DataFrame) -> str:
    """Formats DataFrame into GitHub Flavored Markdown table."""
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers))]) + " |")

    for _, row in df.iterrows():
        row_vals = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def generate_benchmark_markdown(results_csv: str = RESULTS_CSV) -> str:
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    df = pd.read_csv(results_csv)
    return format_markdown_table(df)


def sync_readme(readme_path: str = README_PATH, results_csv: str = RESULTS_CSV):
    """Optionally syncs benchmark tables directly inside README.md if markers are present."""
    if not os.path.exists(readme_path) or not os.path.exists(results_csv):
        return

    table_md = generate_benchmark_markdown(results_csv)
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    start_marker = "<!-- BENCHMARK_TABLE_START -->"
    end_marker = "<!-- BENCHMARK_TABLE_END -->"

    if start_marker in content and end_marker in content:
        prefix = content.split(start_marker)[0] + start_marker + "\n\n"
        suffix = "\n\n" + end_marker + content.split(end_marker)[1]
        new_content = prefix + table_md + suffix
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully synced benchmark table inside: {readme_path}")
    else:
        print("Generated Benchmark Table Markdown:")
        print(table_md)


if __name__ == "__main__":
    sync_readme()
