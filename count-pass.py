'''
brief: count single step pass rate for a bunch of result files
usage: python count-pass.py
input: provide test result file path from stdin

example: find -name '.*Qwen2.5-7B' -type f | python count-pass.py
'''
import sys
import json
import os

K = int(os.environ.get("MAX_ATTEMPTS", "10"))
blacklist = set([
  'auto.',
  'eauto.',
  'simpl.',
  'idtac.',
  'trivial.',
  'Abort.',
  'Admitted.',
  'admit.',
])

if __name__ == "__main__":
    passed = [0 for _ in range(K + 1)]
    total = 0
    # for each result file
    for f in sys.stdin.readlines():
        f = f.strip()
        if len(f) == 0:
            continue
        # for each step
        for test in json.load(open(f)):
            total += 1
            if test["succeeded"] and test["attempts"][-1] not in blacklist:
                # k-th attempt succeeded
                k = len(test["attempts"])
                for i in range(k, K + 1):
                    passed[i] += 1

    if total > 0:
        for i in range(1, K + 1):
            c = passed[i]
            print(f'pass@{i}', f'{c}/{total}', c/total, sep='\t')
    else:
        print("no result available")

