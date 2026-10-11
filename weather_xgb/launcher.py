"""PyInstaller entry point; multiprocessing support precedes Qt imports."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-runtime":
        from pathlib import Path
        import traceback
        folder = Path(sys.argv[2]); folder.mkdir(parents=True, exist_ok=True)
        try:
            from weather_xgb.verification import frozen_check
            frozen_check(folder)
        except Exception:
            (folder / "startup_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            sys.exit(1)
    else:
        from weather_xgb.ui import main
        main()
