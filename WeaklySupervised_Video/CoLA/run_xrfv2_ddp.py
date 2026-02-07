import sys
from scripts.run import main


if __name__ == "__main__":
    sys.argv = ["run.py", "--dataset", "xrfv2", "--model", "cola"] + sys.argv[1:]
    main()
