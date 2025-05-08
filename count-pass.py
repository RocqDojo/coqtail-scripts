import sys
import json
import os

# number of maximum samples allowed, pass@K
K = int(os.environ.get("K", default="5"))

if __name__ == "__main__":
    passed, total = 0, 0
    for f in sys.argv[1:]:
        for test in json.load(open(f)):
            total += 1
            if test["succeeded"] and len(test["attempts"]) <= K:
                passed += 1

    if total > 0:
        print(f"pass@{K} {passed}/{total}", passed / total)
    else:
        print("no result available")
