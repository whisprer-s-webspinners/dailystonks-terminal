from pathlib import Path
import os, sys, subprocess

def main():
    repo = Path(__file__).resolve().parents[1]
    out = repo / "out" / "offline_smoke_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    cmd = [sys.executable, "-m", "dailystonks", "--tier", "black", "--out", str(out), "--offline-synth", "--max-universe", "40"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo), env=env)
    print("OK:", out)

if __name__ == "__main__":
    main()
