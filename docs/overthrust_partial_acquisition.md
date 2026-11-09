# Overthrust: completed bounded raw-plane acquisition

One fixed cross-section of a different named synthetic geological model is now available for future validation. This is **input acquisition**, with no wave or inversion experiment and no accepted physical-accuracy result.

The source is the CC BY 4.0 [SEG/EAGE Overthrust record 4252588](https://zenodo.org/records/4252588), file `overthrust_3D_true_model.h5`, UUID `436595b8-ed4b-4e1d-8991-e63608dcc169`. The whole file is 531,253,092 bytes. Its published MD5 remains a publisher claim: extracting a subset does not independently verify it. Original attribution, license text and HDF5 copyright metadata accompany the saved data.

## What was acquired and checked

Before model values were read, the source-reviewed plan fixed stored dataset `m[:,400,:]`: all 207 first-axis rows and all 801 last-axis samples. The contiguous little-endian float32 layout permits 207 separate 3,204-byte ranges. Every row was retained; no value-dependent selection, conversion, transpose, smoothing or water removal occurred.

The actual process returned exit 0. All **259 requests** completed: 207 model rows, 50 before/after metadata ranges and two primary record responses. The actual response-body total is **686,662 bytes**, including **663,228 model bytes**, under the fixed 8 MiB cap. Range responses required HTTP 206, exact Content-Range/total length, exact body length and identity encoding before reading. Whole-file responses, redirects, retry/resume and large fallbacks were refused. Failed-prefix and negative transport behavior pass 29 manufactured controls; the earlier metadata review passes 487 checks.

The separate saved-byte audit passes **3,186 checks**, with zero integrity failures and one disclosure correction. It verifies all ordered ranges, row and artifact hashes, before/after/cached metadata, and exact byte assembly of the `(207,801)` NPY array. All **165,807 stored values** are finite and positive. It performs no network request, unit conversion or PDE solve.

The correction concerns ETags: the two **record JSON** responses carry `"2"`; the **HDF5 file-range** responses supply no ETag. The producer's blanket no-ETag sentence is too broad and remains preserved in the raw receipt. A record revision ETag does not establish whole-file immutability. The audit verifies saved evidence rather than independently refetching or authenticating the remote byte stream.

The [archive](../results/overthrust_partial_acquisition/README.md) retains the unchanged raw producer output, full request ledger, all row bodies, source/control history, external actual-exit observation, independent audit and primary software context. The ordered raw plane SHA256 is `23a3efe53b0cec5a83832e57b8db07fe0cba3e5d339ad244a9ce3743931e2907`.

## Qualified interpretation and remaining work

The inspected HDF5 metadata directly establishes shape/type/layout, `n=(801,801,207)`, `d=(25,25,25)` and zero origin. It does not directly label the model units or axis names. A pinned [Witte-authored JUDI example](https://github.com/slimgroup/JUDI.jl/blob/655cc4400514fae01bbfb368ce80d96bcd5bb20b/examples/software_paper/generate_data_overthrust_3D.jl) loads this file basename into JUDI Model. Its [model documentation](https://github.com/slimgroup/JUDI.jl/blob/655cc4400514fae01bbfb368ce80d96bcd5bb20b/src/TimeModeling/Types/ModelStructure.jl) describes squared slowness in s²/km², axes `(x,y,z)` and distances in metres. Pinned [HDF5.jl dimension handling](https://github.com/JuliaIO/HDF5.jl/blob/9e003dc6df12b5b97722634e7b4798a69f94bc94/src/dataspaces.jl) reverses the stored dimension order.

Together these support the **conditional software-context interpretation** `plane[z,x] = Julia_m[x+1,401,z+1]`, with 25 m spacing and squared-slowness values. They do not prove FTP-to-Zenodo byte identity, the original writer/package version, an unchanged original CD1 velocity grid, or the provenance of 20 alleged added water rows. Those rows remain included, and no velocity conversion has been executed.

The next physical study still needs a reviewed interpretation/transformation, fixed geometry and acceptance criteria, independent propagation/derivative checks, and boundary/grid/time convergence. A separate synthetic ancestor is not an independent field site. This extraction supplies neither held-out recovery confirmation nor a certification contribution, student-owned interpretation or competition-readiness claim. All four scientific objectives remain open.
