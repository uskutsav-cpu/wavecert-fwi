# Overthrust embedded source: quarantined static review

The pinned SWEEP source is byte-verified and remains quarantined. No acquired
code was executed, no embedded array was decoded, and no wave solver or optimizer
was run. This is an acquisition/provenance result, not geological validation.

Source: [DeepWave-KAUST/sweep, commit 889fccd](https://github.com/DeepWave-KAUST/sweep/blob/889fccd3891e3758052a4db9e39acc5350500236/src/sweep/datasets/overthrust_2d.py).
The 128,769-byte file fits the predeclared 131,072-byte source-body cap. Its Git
blob is `e5489b57c841ed9a40b36a1579d1aae7ff760f3c` and SHA256 is
`f2884b53f1d979e1044068dfefcd8c369143a1a5e54dffff11e27c58bc8d2e98`.
The raw source is preserved unchanged; `STATIC_MASKED_SOURCE.txt` is only a
readable AST rendering with long literals omitted.

| Item | Static evidence and limit |
| --- | --- |
| Presets | Separate `true` and `smooth` embedded literals; both declare shape `(187, 801)` |
| Units/grid | Source constants declare m/s and 25 m spacing |
| Storage | Module description says float16, zlib level 9, then base85. The decoder reads native-endian float16, reshapes with default C order and converts to float32 |
| Loss | Float16 storage is lossy. The author's stated error below 2 m/s was not independently measured and cannot be verified without the original pre-encoding arrays |
| Declared ranges | Metadata gives true 2356.947021484375–6000 and smooth 2575.120361328125–6000 m/s; these are not decoded or verified extrema |
| Decoder transforms | No interpolation, clipping or velocity normalization appears in the decoder. Pre-encoding transforms remain unknown |
| Axes/slice | C-order shape is explicit. The module does not identify the original 3-D slice index, physical origin, extraction orientation or upstream array checksum. Calling its axes depth/lateral is a catalog-based interpretation |
| Portability | Endianness is unspecified in `dtype=np.float16`; an explicit interpretation is needed in any later portable decoder |

The referenced `scripts/encode_npy_to_py.py` is absent from the pinned repository's
complete, untruncated 1,224-entry tree. The small 3-D example scripts explain how
to obtain an original Overthrust cube and generate smoothed cubes, but do not bind
those operations to these embedded literals. Their defaults must not be assigned
to the embedded `smooth` preset. The full archive and HDF5 model were not fetched.

The [original publisher catalog](https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models)
labels the SEG/EAGE model CC BY 4.0 and identifies SEG copyright and the original
authors. A direct page request returned 403 in this review; its search-indexed
publisher text supplied the licensing statement. The pinned SWEEP registry
separately labels its Overthrust demo as derived from a CC BY 4.0 source and cites
Aminzadeh, Brac and Kunz (1997), *SEG/EAGE 3-D Modeling Series No. 1*. Its
[software LICENSE](https://github.com/DeepWave-KAUST/sweep/blob/889fccd3891e3758052a4db9e39acc5350500236/LICENSE)
is MIT. These are distinct software/data notices; MIT is not used to infer data
rights. Exact MIT and registry/catalog sources are retained under `provenance/`.
Any later distribution should preserve both notices, the original citation,
the source commit and a modification/quantization description. The original
archive's own license file was not acquired here.

A bounded local search found Overthrust only in the repository's ancestry notes;
the visible external model cache contains `mar_big_117_567.bin`. This does not
prove that no earlier process or differently named adaptation touched Overthrust.
Treat this as a candidate derived demo until the broader prior-use ledger is
resolved. One synthetic Overthrust ancestor, its smooth version and any crops do
not become independent sites or trials.

Before model decoding, root should review the unresolved source-slice,
endianness, quantization-reference and prior-use questions and explicitly assign
a derived-demo role. No native-original, untouched-confirmation, physical
acceptance, novelty or student-authorship claim follows from this review.
