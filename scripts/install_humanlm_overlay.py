"""Install and verify the DECO/PDN overlay in the pinned HumanLM repository."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

PINNED_COMMIT = "6a7dbd3f143fc0a9af599ed7a458fc503341f846"


def run(command: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=check, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository", type=Path, help="Path to the verl-recipe-humanlm checkout")
    parser.add_argument("--allow-other-commit", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verify applicability without changing files")
    args = parser.parse_args()

    repository = args.repository.resolve()
    root = Path(__file__).resolve().parents[1]
    patch = root / "training" / "patches" / "humanlm-reward-manager.patch"
    metric = root / "training" / "humanlm_overlay" / "metrics" / "state_reward_on_response.py"
    destination = repository / "humanlm" / "metrics" / "state_reward_on_response.py"

    if not (repository / ".git").exists() or not destination.exists():
        raise SystemExit(f"Not a HumanLM repository checkout: {repository}")
    commit = run(["git", "rev-parse", "HEAD"], repository).stdout.strip()
    if commit != PINNED_COMMIT and not args.allow_other_commit:
        raise SystemExit(f"Expected {PINNED_COMMIT}, found {commit}")

    check = run(["git", "apply", "--check", str(patch)], repository, check=False)
    if check.returncode != 0:
        raise SystemExit(f"Patch cannot be applied cleanly:\n{check.stderr}")
    if args.check:
        print("The reward-manager patch applies cleanly.")
        return

    run(["git", "apply", str(patch)], repository)
    shutil.copy2(metric, destination)
    print(f"Installed the overlay in {repository}")


if __name__ == "__main__":
    main()

