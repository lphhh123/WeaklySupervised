import sys
from scripts.run import main


if __name__ == "__main__":
    sys.argv = ["run.py", "--dataset", "hangtime", "--model", "rskp"] + sys.argv[1:]
    main()
