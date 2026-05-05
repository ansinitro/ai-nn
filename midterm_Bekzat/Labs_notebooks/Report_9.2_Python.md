# Section 9.2: Python Programming Foundations

**Student:** Sundetkhan Bekzat

## Purpose

Section 9.2 builds the programming base required for later machine learning work. The notebooks cover Python syntax, core containers, loops, functions, object-oriented design, file processing, regular expressions, decorators, and exception handling.

## Completed Labs

- `9.2.1`: data types, deterministic branch checks, loops, Fibonacci generation, class-level state, and standard library inspection.
- `9.2.2`: text, CSV, and JSON file operations using temporary folders; a small database-like adapter replaces an external MySQL dependency.
- `9.2.3`: regex extraction and validation, iterable handling, and a timing decorator.
- `9.2.4`: treasury ledger with transactions, balances, category totals, formatted statements, and safe input parsing.

## Engineering Notes

The implementation avoids interactive `input()` calls so every notebook can be executed by `nbconvert`. External services are replaced with local adapters where the lab concept is more important than a real server connection.

## Result

The section produces reproducible terminal outputs and reusable utility patterns that prepare the later tabular and text-processing notebooks.
