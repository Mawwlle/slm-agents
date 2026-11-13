---
name: python-function-tdd
description: How to write a Python function that must pass given assert tests, check it and finish.
---
1. Read the example test first: it fixes the function name, the argument order and the return type. Use exactly that name.
2. Write the whole solution.py with write_solution: imports at the top, one function (plus helpers), no prints, no input().
3. Call run_tests. If it fails, read the assertion: compare the expected value with what your function returns (use the python tool to print it), fix, and run_tests again.
4. Think about edge cases the hidden tests may use: empty input, one element, negative numbers, ties, case of letters.
5. Return the type the test compares with (list vs tuple, int vs float, str vs bool).
6. When run_tests passes, call finish. Do not rewrite a passing solution.
