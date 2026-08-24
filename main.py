from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        print(
            "Dependencies are missing. Run: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    app = QApplication(sys.argv)
    app.setApplicationName("Multi-Camera 3D Crop Studio")
    try:
        from realsense_cropper.main_window import MainWindow

        window = MainWindow()
        window.show()
        return app.exec()
    except ImportError as exc:
        QMessageBox.critical(
            None,
            "Missing dependency",
            f"A dependency is missing:\n{exc}\n\n"
            "Install the project dependencies from requirements.txt. "
            "See README.md for Windows and Ubuntu instructions.",
        )
        return 2
    except Exception as exc:  # keep startup failures visible outside a terminal
        QMessageBox.critical(None, "Startup error", str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
