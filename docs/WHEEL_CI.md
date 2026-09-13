# Wheel compatibility CI

The prepared workflow builds wheel and source archives on Ubuntu 24.04 x86_64,
Windows 2022 x86_64 and macOS 15 arm64 with CPython 3.10–3.13: 12 jobs. GitHub
Actions are pinned to full commits. The earlier workflow passed actionlint 1.7.12
locally. This candidate expands the explicit test selection to include ROI/scale
parity and persistent Format lifecycle tests; lint and execution for that edit
remain pending while both benchmark hosts are reserved. No hosted run or platform
result is claimed yet.

Each job installs the built wheel into a new virtual environment, imports it
without OpenCV or Ultralytics installed, and verifies wheel RECORD plus every
installed runtime/data byte. All notice files must match the repository, wheel,
sdist and installed package. The NumPy-only mask smoke is followed by the core
OpenCV parity suite, including ROI/scale boundaries and scratch reuse. Python
3.11–3.13 additionally run framework integration, training and persistent Format
rebuild/worker-reset tests against the pinned Ultralytics commit and numerical
dependencies.
Python 3.10 has core-only coverage, using NumPy 2.2.6; it is not a validated
framework-adapter profile. Reports and artifacts are retained for 14 days.

The build uses private static OpenCV libraries. Their MSVC runtime selection is
explicitly dynamic, matching the extension's normal CMake runtime selection;
this prevents OpenCV's static-library default from selecting a different CRT.
The Windows link and downstream DLL requirements still need actual validation.
The benchmark worktree and installed benchmark extensions are unchanged by this
build configuration update.

This is a compatibility workflow, not a portable release pipeline. Linux wheels
are not yet manylinux-repaired. A macOS 11 deployment target does not prove
execution on macOS 11. Windows tests on a developer runner do not prove a clean
end-user installation has every required runtime DLL. Platform dependency and
notice audits, portable wheel repair, clean-machine testing, release-candidate
benchmarks and actual publication remain release gates. Repository creation and
hosted execution are pending the user's public/private repository choice.

The persistent candidate workflow also passed actionlint 1.7.12 on Linux after
the ROI/persistent test selections were added. Its exact hash is retained in
[the candidate lint receipt](validation/mask-persistent-linux-v1-lint.json).
This is workflow syntax evidence only; the hosted matrix remains unexecuted.
