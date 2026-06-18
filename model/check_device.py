from __future__ import annotations

import json

from model.device import device_report, select_device


def main() -> None:
    report = device_report()
    report["auto_selected_device"] = str(select_device("auto"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
