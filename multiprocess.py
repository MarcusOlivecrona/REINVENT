#!/usr/bin/env python

import importlib
import sys

scoring_function = sys.argv[1]
func = getattr(importlib.import_module("scoring_functions"), scoring_function)()

while True:
    smile = sys.stdin.readline().rstrip()
    try:
        score = float(func(smile))
    except:
        score = 0.0
    sys.stdout.write(" ".join([smile, str(score), "\n"]))
    sys.stdout.flush()



