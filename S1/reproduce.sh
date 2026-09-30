#!/usr/bin/env bash
# Reproduces what S1 can reproduce without the mission recordings: the contract tests, Tables 3-5 and Figure 1.
set -e; cd "$(dirname "$0")"
echo "== 1. contract tests of the reference implementation"; g++ -std=c++11 -Wall -Wextra -Wpedantic -Werror -O2 -Icode code/test_contextmemory.cpp -o /tmp/test_contextmemory && /tmp/test_contextmemory
echo "== 2. replay harness compiles (running it needs the recordings; see README)"; g++ -std=c++11 -Wall -Wextra -Wpedantic -Werror -O2 -Icode code/replay_stageA.cpp -o /tmp/replay_stageA && echo "   ok"
echo "== 3. Tables 3, 4, 5 from the result files"; python3 code/reproduce_tables.py
echo "== 4. Figure 1"; python3 code/make_figure.py
