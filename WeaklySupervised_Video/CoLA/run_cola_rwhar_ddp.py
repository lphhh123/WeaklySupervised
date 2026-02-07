import sys
from scripts.run import main


if __name__ == "__main__":
    sys.argv = ["run.py", "--dataset", "rwhar", "--model", "cola"] + sys.argv[1:]
    main()
