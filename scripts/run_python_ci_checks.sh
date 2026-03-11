#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROJECT=""
MODE="all"
USE_VENV="${USE_VENV:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)
      PROJECT="${2:-}"
      shift 2
      ;;
    --lint-only)
      MODE="lint"
      shift
      ;;
    --test-only)
      MODE="test"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

discover_projects() {
  local d
  for d in "$ROOT_DIR"/*; do
    [[ -d "$d" ]] || continue
    local name
    name="$(basename "$d")"
    [[ "$name" == .* ]] && continue
    if [[ -f "$d/pyproject.toml" || -f "$d/setup.py" || -f "$d/requirements.txt" ]]; then
      printf '%s\n' "$name"
    fi
  done
}

setup_python() {
  local project="$1"
  if [[ "$USE_VENV" == "1" ]]; then
    local venv_dir="$ROOT_DIR/.git/.husky-venvs/$project"
    python3 -m venv "$venv_dir"
    printf '%s\n' "$venv_dir/bin/python"
  else
    if command -v python >/dev/null 2>&1; then
      printf '%s\n' "python"
    else
      printf '%s\n' "python3"
    fi
  fi
}

install_deps() {
  local py="$1"
  "$py" -m pip install --upgrade pip
  if [[ -f pyproject.toml || -f setup.py ]]; then
    "$py" -m pip install ".[dev]" || "$py" -m pip install .
  fi
  if [[ -f requirements-dev.txt ]]; then
    "$py" -m pip install -r requirements-dev.txt
  fi
  if [[ -f requirements.txt ]]; then
    "$py" -m pip install -r requirements.txt
  fi
}

run_lint() {
  local py="$1"
  if "$py" -m ruff --version >/dev/null 2>&1; then
    "$py" -m ruff check --exclude=tests/golden .
  elif "$py" -m flake8 --version >/dev/null 2>&1; then
    "$py" -m flake8 .
  else
    echo "No supported linter installed (expected ruff or flake8)." >&2
    return 1
  fi

  if grep -q '^\[tool\.mypy\]' pyproject.toml 2>/dev/null; then
    if "$py" -m mypy --version >/dev/null 2>&1; then
      "$py" -m mypy .
    else
      echo "mypy is configured for this project but is not installed." >&2
      return 1
    fi
  fi
}

run_tests() {
  local py="$1"
  if "$py" -m pytest --version >/dev/null 2>&1; then
    "$py" -m pytest
  else
    echo "pytest is not installed." >&2
    return 1
  fi
}

main() {
  local projects=()
  if [[ -n "$PROJECT" ]]; then
    projects=("$PROJECT")
  else
    while IFS= read -r p; do
      projects+=("$p")
    done < <(discover_projects)
  fi

  if [[ "${#projects[@]}" -eq 0 ]]; then
    echo "No Python projects found at repository root."
    exit 0
  fi

  local p py
  for p in "${projects[@]}"; do
    if [[ ! -d "$ROOT_DIR/$p" ]]; then
      echo "Project directory not found: $p" >&2
      exit 1
    fi
    echo "==> $p"
    pushd "$ROOT_DIR/$p" >/dev/null
    py="$(setup_python "$p")"
    install_deps "$py"
    if [[ "$MODE" == "all" || "$MODE" == "lint" ]]; then
      run_lint "$py"
    fi
    if [[ "$MODE" == "all" || "$MODE" == "test" ]]; then
      run_tests "$py"
    fi
    popd >/dev/null
  done
}

main
