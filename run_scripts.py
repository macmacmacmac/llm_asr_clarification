import importlib
import argparse
import sys
from llm_asr_clarification import get_logger


def main():
    # print("\nReceived the following arguments:\n|---->", " ".join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scripts",
        nargs="+",                   # <── allow multiple
        required=True,
        help="scripts to run"
    )
    args, unknown = parser.parse_known_args()

    for script in args.scripts:
        module_name = f"llm_asr_clarification.scripts.{script}"
        script_module = importlib.import_module(module_name)

        if not hasattr(script_module, "run"):
            raise ValueError(f"Script '{script}' must have a run(args) function.")

        # Each experiment receives same unknown args (or you could customize)
        script_module.run(unknown)


if __name__ == "__main__":
    main()
