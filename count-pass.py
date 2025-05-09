import sys
import json
import os

K = int(os.environ.get("MAX_ATTEMPTS", "64"))

if __name__ == "__main__":
    passed = [0 for _ in range(K)]
    total = 0
    for f in sys.argv[1:]:
        for test in json.load(open(f)):
            total += 1
            if test["succeeded"]:
                # k-th attempt succeeded
                k = len(test["attempts"])
                for i in range(k, K):
                    passed[i] += 1

    if total > 0:
        rate = [i / total for i in passed]
        print(rate)
    else:
        print("no result available")
