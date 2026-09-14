# Current packet wheel: M2 installed suite passed

A fresh build from `0edc3bb3ba5fa211d54bb8d69da2490aa7695ed5` passed
**366 installed tests, no failures/skips, in 274.28 seconds**. Pytest exited
naturally with code zero. All 16 build, installation, collection, test and
byte-audit commands exited zero on their first execution.

## Environment and artifact checks

The wheel targets CPython 3.12 / macOS arm64 with deployment target 11.0.
The Mach-O load command also records minimum macOS 11.0. Actual execution was
on **macOS 26.6.2, Python 3.12.13**; older macOS execution remains unverified.

A clean NumPy-only environment passed a native rasterization smoke test without
Python OpenCV or Ultralytics. Wheel RECORD, installed bytes and all nine notice
files matched source/wheel/sdist. The framework environment passed `pip check`;
47 normalized non-mask/non-pip dependency entries matched the previous M2
environment. These include NumPy 2.4.4, Pillow 12.1.1, OpenCV 4.13.0.92,
Torch 2.10.0 and the previously pinned Ultralytics source. Exact local Ultralytics
and dataset archives are preserved. Installing the dataset wheel here does not
constitute a new dataset-suite or combined real-training result.

The extension links only system Accelerate, libc++ and libSystem libraries;
`nm -gU` reports only `_PyInit__native`. All **573 selected compiled
OpenCV/KleidiCV/pybind11 source/header files** match the hash-bound notice
provenance. System, toolchain, Python headers and generated files remain outside
that collection's scope.

The first read-only inspection mistakenly used the compact notice manifest in
place of its full provenance and raised `KeyError: source_files`. Its failed
receipt is preserved. The corrected inspection uses the full provenance hash
already bound by the compact manifest. No build or tests were repeated.

## Functional scope

| Test group | Passed |
| --- | ---: |
| Core parity and resize ROI | 168 |
| Framework integration and CPU training | 19 |
| Persistent formatter and shared collator | 48 |
| Synthetic benchmark/coordinator/auditor checks | 131 |
| Total | 366 |

Worker tests exercise setup, first-batch and full-epoch close-mosaic resets,
overlap on/off and zero/two workers. They check old/replacement worker exit codes
and exact post-reset batches. Their reference retains original collation
computation with test-only normalized packet transport, as described in the
[Linux packet qualification](SHARED_PACKET_VALIDATION.md). Four CPU training
cases compare loss, gradients and parameter updates.

This M2 Torch runtime exposes only `file_system`; the suite used that default.
Pytest exited naturally, and the subsequent process-table observation found no
recorded command PIDs, immediate children or identifiable private-runtime
shared-memory managers remaining. This is bounded functional/exit evidence.
It does not measure named-storage retention, establish long-run reliability or
resolve the separate [Linux failure](SHARED_TRANSPORT_LIFETIME.md). Repeated
fresh-process stress, storage-lifetime measurements, real-data M2 training,
other Python versions and hosted platform execution remain open. No speedup
or memory reduction is inferred from this suite.

## Preserved evidence

The [43-member archive](../bench/results/mask-packet-m2-v1-evidence.tar.gz) contains
frozen selected source, wheel/sdist, raw logs/JUnit, dependency/environment
identities, compiler graph/build metadata, binary inspection, scripts and exit
observation. Every member was read back and hashed against the
[manifest](validation/mask-packet-m2-v1-preservation.json).

- Archive: 5,718,671 bytes, SHA-256
  `653404406959a3634125c94e6a213af75894fc374d5c4bb40706be8d06c4e041`.
- Wheel: `414caeb9148a2629ebcc15f1f3b3eccc0dacb8b673a066cdb1d9c8560ea5d86f`.
- Extension: `1207e38c6286b7a31c9b7314cb1db3fafb1e1e030837e724024b52c76e9c43b0`.
- Source tar: `cb1853b13e8dc9482216aecd391381c9c06d156d4b122983db959102689e162e`.

See [qualification](validation/mask-packet-m2-v1-qualification.json),
[JUnit](validation/mask-packet-m2-v1-tests.xml),
[native inspection](validation/mask-packet-m2-v1-native-inspection-v2.json) and
[exit observation](validation/mask-packet-m2-v1-exit-observation.json).
