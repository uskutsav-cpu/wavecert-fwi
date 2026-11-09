# First remote replay regression: mixed outcome

Commit c197a43f5774f4a320a1de19bad85f8b9f4cea1a, workflow42466790405:

- Learned/data job128688043649 passed, including all three BP traces and six derivatives, with563 tests passed/4 skipped/1 warning.
- Python3.12 matrix job128688043908 failed because the new replay subprocess returned2. The original test captured its output without exposing the saved comparison details. The exact log is retained.
- Python3.10/3.11 jobs were cancelled by matrix fail-fast and receive no numerical credit.

The passing/failing jobs report the same Python3.12.16/NumPy2.5.4/Linux-x86_64 labels. The cause is unresolved. A diagnostic-only test patch will expose per-case errors and saved model differences while keeping every numerical gate, frozen replay core/protocol, failure result and denominator unchanged. No tolerance relaxation or native rerun is authorized by these records.

The successful six known-fixture reports are retained in the JSON; they are not six new research studies or new geological samples. Source-only triage suspects exact model-byte/proxy checks, but CI diagnostics must establish the cause.
