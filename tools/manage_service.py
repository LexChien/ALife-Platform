"""Install/manage a loopback-only systemd user service without sudo."""
import argparse
from pathlib import Path
import subprocess


UNIT = "alife-platform.service"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["install", "start", "stop", "restart", "status", "uninstall"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    unit = Path.home() / ".config/systemd/user" / UNIT
    if args.action == "install":
        unit.parent.mkdir(parents=True, exist_ok=True)
        # systemd quoted values escape literal percent specifiers in user paths.
        directory = str(root).replace("%", "%%").replace('"', '\\"')
        unit.write_text(
            "[Unit]\nDescription=ALife Platform local API\nAfter=network.target\n\n"
            f"[Service]\nType=simple\nWorkingDirectory=\"{directory}\"\n"
            f"ExecStart=/usr/bin/python3 \"{directory}/tools/run_python.py\" -m uvicorn apps.service:app --host 127.0.0.1 --port 8765 --workers 1\n"
            "Restart=on-failure\nRestartSec=5\nTimeoutStopSec=15\nKillMode=control-group\n"
            "Environment=PYTHONUNBUFFERED=1\nEnvironment=TOKENIZERS_PARALLELISM=false\n"
            "\n[Install]\nWantedBy=default.target\n")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "--now", UNIT], check=True)
        print("ALife API: http://127.0.0.1:8765/docs")
    elif args.action == "uninstall":
        subprocess.run(["systemctl", "--user", "disable", "--now", UNIT], check=True)
        unit.unlink(missing_ok=True)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    else:
        subprocess.run(["systemctl", "--user", args.action, UNIT], check=True)


if __name__ == "__main__":
    main()
