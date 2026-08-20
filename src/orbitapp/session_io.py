"""Session save/load: two HDF5 files per session.

- **Pipeline file** (always written): the reproducibility "recipe" --
  ordered committed steps with their exact parameters, correlation-based
  ROIs' seed_loc/threshold in the order they were added, batch methods'
  (PCA-ICA/CNMF) shared parameters, and ROI Validation's last-used
  auto-classify/Run-SEUDO parameters. No movie or ROI mask/trace arrays
  -- small, portable, meant for provenance/reproducibility.
- **Output file** (written only on a "full save"): the actual derived
  results -- ROI masks/traces/spike_traces/neuropil_traces, each
  committed stage's headline QC metrics, and ROI Validation's actual
  per-transient classification/is_artifact/times.

Movies themselves are deliberately never saved (by explicit design
decision) -- reloading a session re-associates with the original movie
via its recorded path (AppState.data_path) rather than duplicating
(often huge) pixel data.

Both files use small JSON-encoded string attributes for nested
dicts/lists (params, metrics, seed_loc) -- a standard, simple way to
carry arbitrary metadata in HDF5 without inventing a bespoke
group-per-key encoding, and real HDF5 datasets for the numeric arrays
(masks, traces, transient times).
"""

from __future__ import annotations

import json

import numpy as np

from .state import AppState, PipelineStep, ROI


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


def save_pipeline(state: AppState, roi_validation_params: dict, path) -> None:
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["data_path"] = state.data_path or ""
        f.attrs["pipeline"] = _dumps(state.pipeline)

        steps_group = f.create_group("steps")
        for i, step in enumerate(state.steps):
            g = steps_group.create_group(str(i))
            g.attrs["stage"] = step.stage
            g.attrs["label"] = step.label
            g.attrs["params"] = _dumps(step.params)

        rois_group = f.create_group("source_extraction_rois")
        for i, roi in enumerate(state.rois):
            g = rois_group.create_group(str(i))
            g.attrs["id"] = roi.id
            g.attrs["source_method"] = roi.source_method
            g.attrs["status"] = roi.status
            g.attrs["seed_loc"] = _dumps(list(roi.seed_loc) if roi.seed_loc is not None else None)
            g.attrs["params"] = _dumps(roi.params or {})

        f.attrs["roi_validation"] = _dumps(roi_validation_params)


def load_pipeline(path) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        steps = []
        steps_group = f["steps"]
        for i in sorted(steps_group.keys(), key=int):
            g = steps_group[i]
            steps.append(PipelineStep(stage=g.attrs["stage"], label=g.attrs["label"], params=json.loads(g.attrs["params"])))

        source_extraction_rois = []
        rois_group = f["source_extraction_rois"]
        for i in sorted(rois_group.keys(), key=int):
            g = rois_group[i]
            seed_loc = json.loads(g.attrs["seed_loc"])
            source_extraction_rois.append(dict(
                id=int(g.attrs["id"]), source_method=g.attrs["source_method"], status=g.attrs["status"],
                seed_loc=tuple(seed_loc) if seed_loc is not None else None, params=json.loads(g.attrs["params"]),
            ))

        return dict(
            data_path=f.attrs["data_path"] or None, pipeline=json.loads(f.attrs["pipeline"]), steps=steps,
            source_extraction_rois=source_extraction_rois,
            roi_validation=json.loads(f.attrs["roi_validation"]) if "roi_validation" in f.attrs else {},
        )


def save_output(state: AppState, roi_validation_results: list[dict] | None, path) -> None:
    import h5py

    with h5py.File(path, "w") as f:
        rois_group = f.create_group("rois")
        for i, roi in enumerate(state.rois):
            g = rois_group.create_group(str(i))
            g.attrs["id"] = roi.id
            g.attrs["source_method"] = roi.source_method
            g.attrs["status"] = roi.status
            g.create_dataset("mask", data=roi.mask, compression="gzip")
            g.create_dataset("trace", data=roi.trace, compression="gzip")
            if roi.spike_trace is not None:
                g.create_dataset("spike_trace", data=roi.spike_trace, compression="gzip")
            if roi.neuropil_trace is not None:
                g.create_dataset("neuropil_trace", data=roi.neuropil_trace, compression="gzip")

        steps_group = f.create_group("steps")
        for i, step in enumerate(state.steps):
            g = steps_group.create_group(str(i))
            g.attrs["label"] = step.label
            g.attrs["metrics"] = _dumps(step.metrics)

        if roi_validation_results is not None:
            rv_group = f.create_group("roi_validation_rois")
            for i, entry in enumerate(roi_validation_results):
                g = rv_group.create_group(str(i))
                g.create_dataset("times", data=np.asarray(entry["times"], dtype=int))
                g.create_dataset("classification", data=np.asarray(entry["classification"], dtype=float))
                g.attrs["is_artifact"] = bool(entry["is_artifact"])


def load_output(path) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        rois = []
        rois_group = f["rois"]
        for i in sorted(rois_group.keys(), key=int):
            g = rois_group[i]
            rois.append(ROI(
                id=int(g.attrs["id"]), mask=g["mask"][()].astype(bool), trace=g["trace"][()],
                source_method=g.attrs["source_method"], status=g.attrs["status"],
                spike_trace=g["spike_trace"][()] if "spike_trace" in g else None,
                neuropil_trace=g["neuropil_trace"][()] if "neuropil_trace" in g else None,
            ))

        metrics_by_label = {}
        steps_group = f["steps"]
        for i in sorted(steps_group.keys(), key=int):
            g = steps_group[i]
            metrics_by_label[g.attrs["label"]] = json.loads(g.attrs["metrics"])

        roi_validation_results = None
        if "roi_validation_rois" in f:
            roi_validation_results = []
            rv_group = f["roi_validation_rois"]
            for i in sorted(rv_group.keys(), key=int):
                g = rv_group[i]
                roi_validation_results.append(dict(
                    times=g["times"][()], classification=g["classification"][()], is_artifact=bool(g.attrs["is_artifact"]),
                ))

        return dict(rois=rois, metrics_by_label=metrics_by_label, roi_validation_results=roi_validation_results)
