# Batched common-model neural objective and gradient

`TrainedFNOWavefieldSurrogate.survey_objective_and_gradient(m, blocks, receivers)`
is an optional inference API. It accepts source/frequency `SurveyBlock` entries
with one shared squared-slowness vector and common receivers. The FNO processes
the survey batch in one forward invocation. Autograd differentiates the summed
misfit with respect to that same model vector; expanding the shared tensor
accumulates every block's contribution. Repeated receiver samples retain their
multiplicity in both the objective and adjoint.

The returned values are the summed objective, common-model gradient,
`(n_blocks,n_grid)` complex wavefields, and `(n_blocks,n_receivers)` complex
residuals in input order. This follows the existing point-source-mask and
float32 adapter semantics. It does not change optimizer defaults, integrate an
exact directional oracle, or define a new certificate. Larger batches require
more activation memory; automatic chunking is not implemented.

Tests compare the actual batched gradient and receiver outputs with the serial
API, verify a forward-JVP/reverse-VJP directional identity and independent
objective finite differences, and check repeated-receiver adjoint accumulation.
They also verify one batched FNO invocation rather than an internal block loop.
Float32 kernel/reduction differences are explicitly tolerated and measured.

## Invocation benchmark

From the repository root:

```
PYTHONPATH=src MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_batched_neural.py
```

The benchmark uses the unchanged `fno_parent_grouped.pt` checkpoint and one
deterministic synthetic 24×24 model. Batch sizes 4, 12, and 40 each include
different sources and frequencies, common receivers, and observations generated
by the reference Helmholtz solver outside the timed work. This engineering
control does not measure geological generalization or surrogate accuracy.

Each method, batch size, and process repetition runs in a fresh sequential
subprocess with one Torch/BLAS thread. Checkpoint loading, process startup,
observation synthesis, JSON serialization, and correctness comparisons are
untimed. The first timed call is the first neural objective/gradient invocation
in that subprocess. The operating-system and hardware caches are not flushed;
this is an application first-call measurement. One further untimed call warms
the method, followed by five timed calls. Three fresh processes per method and
batch size are used by default, with alternating method order.

Both paths return their complete normal API outputs, including CPU NumPy
conversion. The serial path sums the existing per-block objectives/gradients
and stacks its wavefields/residuals. The batched path sums one survey loss.
No hooks are added to the timed model. Full gradients and actual complex
receiver predictions/residuals are compared; hashes alone are not treated as
numerical agreement. Raw worker outputs, first-call/warm timings, source and
checkpoint hashes, and runtime/thread metadata are preserved in
`results/batched_neural_controls/`.

This benchmark measures API invocation times. It does not establish a full
inversion acceleration, useful neural directions, lower total physics work,
memory scalability, or time to a reconstruction target. Those require a
separate complete inversion comparison with the equally optimized exact
baseline and all online verification costs included.
