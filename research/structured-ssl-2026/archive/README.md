# Historical evidence

The current implementation and supported commands are V3 only. `v1/` and
`v2/` preserve earlier documents and diagnostic artifacts byte for byte.
Their relative paths, status snapshots and architecture claims describe the
original revisions; use the [manifest](manifest.json) to map original paths.
Executable historical scripts are available at Git commit
`5475c323137381d4f75ff533d74434a81857d719`.

`migrations/policy_handoff.py` records the one-off 70% evaluation-policy handoff.
The running V3 coordinator executes its separate immutable copy under `runs/`.
It is not a supported launch command. The suspended V1 coordinator was retired
following the user's V3-only direction; completed results and source exports
remain preserved. The V3 primary, two controls and two further seeds continue.
