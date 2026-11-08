# Paid-merit software integration verification

This is AI-authored software support, not a student final research design or a
project-data speed/trajectory result. The four added modules retain paid LU
fields, solve missing fields with a norm-free coarse/DST GMRES reference, bound
cached-field gradient errors, and account for budget-limited paid trajectories.
Original 156 production files are byte-identical to be56a4a.

Root verification records a 560-test full-suite pass before the final memory-safe
GMRES default and test-import cleanup; 89 focused tests pass afterward with
warnings as errors. The added regression caps default Krylov storage at 50
vectors while retaining true-residual checks and explicit override. All cost,
novelty, floating-point and student-ownership limits remain explicit. No project
benchmark was executed by this integration. Separate AI review is not human
reproduction. See docs/cached_merit_scope.md and docs/paid_descent_design.md.
