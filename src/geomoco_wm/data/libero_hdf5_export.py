"""Export official LIBERO HDF5 demos into GeoMoCo-WM episode/window records.

The first exporter keeps RGB data in-place by storing HDF5 references and frame
indices. It materializes only lightweight numeric targets: current context
indices, future EEF states, coordinate deltas from the anchor EEF state, and
future action chunks.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import h5py
except ImportError:  # pragma: no cover - h5py is optional at import time
    h5py = None

from geomoco_wm.data.libero_hdf5_inspect import discover_libero_hdf5_files


SCHEMA_VERSION = "libero_hdf5_window_v0"
DEFAULT_CAMERA_KEYS = ("agentview_rgb", "eye_in_hand_rgb")
DEFAULT_LIBERO_SUITE_NAMES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


class LiberoHdf5ExportError(ValueError):
    """Raised when LIBERO HDF5 window export fails validation."""


@dataclass(frozen=True)
class LiberoEpisodeRecord:
    schema_version: str
    episode_id: str
    task_id: str
    suite_name: str
    source_file: str
    demo_name: str
    num_frames: int
    camera_refs: dict[str, dict[str, Any]]
    numeric_refs: dict[str, str]
    action_dim: int
    ee_dim: int
    gripper_dim: int
    joint_dim: int

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def validate(self) -> None:
        _ensure_non_empty(self.schema_version, "episode.schema_version")
        _ensure_non_empty(self.episode_id, "episode.episode_id")
        _ensure_non_empty(self.task_id, "episode.task_id")
        _ensure_non_empty(self.suite_name, "episode.suite_name")
        _ensure_non_empty(self.source_file, "episode.source_file")
        _ensure_non_empty(self.demo_name, "episode.demo_name")
        if self.num_frames <= 0:
            raise LiberoHdf5ExportError("episode.num_frames must be positive")
        if self.action_dim <= 0 or self.ee_dim <= 0 or self.gripper_dim <= 0 or self.joint_dim <= 0:
            raise LiberoHdf5ExportError("episode dimensions must be positive")
        if not self.camera_refs:
            raise LiberoHdf5ExportError("episode.camera_refs must be non-empty")
        if not self.numeric_refs:
            raise LiberoHdf5ExportError("episode.numeric_refs must be non-empty")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiberoEpisodeRecord":
        record = cls(**payload)
        record.validate()
        return record


@dataclass(frozen=True)
class LiberoWindowRecord:
    schema_version: str
    window_id: str
    episode_id: str
    task_id: str
    suite_name: str
    source_file: str
    demo_name: str
    context_start: int
    context_end: int
    anchor_index: int
    future_start: int
    future_end: int
    action_start: int
    action_end: int
    context_frame_indices: list[int]
    future_frame_indices: list[int]
    camera_keys: list[str]
    anchor_ee_state: list[float]
    future_ee_states: list[list[float]]
    future_delta_ee_states: list[list[float]]
    action_chunk: list[list[float]]
    current_gripper_state: list[float]
    current_joint_state: list[float]

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def validate(self) -> None:
        _ensure_non_empty(self.schema_version, "window.schema_version")
        _ensure_non_empty(self.window_id, "window.window_id")
        _ensure_non_empty(self.episode_id, "window.episode_id")
        _ensure_non_empty(self.task_id, "window.task_id")
        _ensure_non_empty(self.suite_name, "window.suite_name")
        _ensure_non_empty(self.source_file, "window.source_file")
        _ensure_non_empty(self.demo_name, "window.demo_name")
        if self.context_start < 0 or self.context_end <= self.context_start:
            raise LiberoHdf5ExportError("window context span is invalid")
        if self.future_start != self.anchor_index + 1:
            raise LiberoHdf5ExportError("window future must start after anchor_index")
        if self.future_end <= self.future_start:
            raise LiberoHdf5ExportError("window future span is invalid")
        if self.action_start != self.anchor_index:
            raise LiberoHdf5ExportError("window action chunk must start at anchor_index")
        if self.action_end - self.action_start != self.future_end - self.future_start:
            raise LiberoHdf5ExportError("window action horizon must match future horizon")
        if self.context_frame_indices != list(range(self.context_start, self.context_end)):
            raise LiberoHdf5ExportError("window context_frame_indices do not match context span")
        if self.future_frame_indices != list(range(self.future_start, self.future_end)):
            raise LiberoHdf5ExportError("window future_frame_indices do not match future span")
        _ensure_float_list(self.anchor_ee_state, "window.anchor_ee_state")
        _ensure_float_rows(self.future_ee_states, "window.future_ee_states")
        _ensure_float_rows(self.future_delta_ee_states, "window.future_delta_ee_states")
        _ensure_float_rows(self.action_chunk, "window.action_chunk")
        _ensure_float_list(self.current_gripper_state, "window.current_gripper_state")
        _ensure_float_list(self.current_joint_state, "window.current_joint_state")
        horizon = self.future_end - self.future_start
        if len(self.future_ee_states) != horizon:
            raise LiberoHdf5ExportError("window future_ee_states length must match horizon")
        if len(self.future_delta_ee_states) != horizon:
            raise LiberoHdf5ExportError("window future_delta_ee_states length must match horizon")
        if len(self.action_chunk) != horizon:
            raise LiberoHdf5ExportError("window action_chunk length must match horizon")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiberoWindowRecord":
        record = cls(**payload)
        record.validate()
        return record


@dataclass(frozen=True)
class LiberoExportSummary:
    input_path: str
    output_dir: str
    suite_name: str
    context_len: int
    horizon: int
    stride: int
    num_files: int
    num_episodes: int
    num_windows: int
    num_frames: int
    dropped_short_episodes: int
    tasks: dict[str, int]
    paths: dict[str, str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LiberoSuiteCollectionExportSummary:
    input_root: str
    output_dir: str
    suite_names: list[str]
    context_len: int
    horizon: int
    stride: int
    num_suites: int
    num_files: int
    num_episodes: int
    num_windows: int
    num_frames: int
    dropped_short_episodes: int
    tasks: dict[str, int]
    suites: dict[str, dict[str, Any]]
    paths: dict[str, str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_libero_hdf5_windows(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    suite_name: str | None = None,
    context_len: int = 2,
    horizon: int = 16,
    stride: int = 4,
    max_files: int | None = None,
    max_demos_per_file: int | None = None,
    max_windows: int | None = None,
    camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS,
) -> LiberoExportSummary:
    """Export episode references and future-motion/action windows."""

    _validate_positive(context_len, "context_len")
    _validate_positive(horizon, "horizon")
    _validate_positive(stride, "stride")
    if max_files is not None:
        _validate_positive(max_files, "max_files")
    if max_demos_per_file is not None:
        _validate_positive(max_demos_per_file, "max_demos_per_file")
    if max_windows is not None:
        _validate_positive(max_windows, "max_windows")
    if h5py is None:
        raise LiberoHdf5ExportError("h5py is required to export LIBERO HDF5 windows")

    resolved_input_path = Path(input_path).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    files = discover_libero_hdf5_files(resolved_input_path)
    if max_files is not None:
        files = files[:max_files]
    effective_suite_name = suite_name or (
        resolved_input_path.name
        if resolved_input_path.is_dir()
        else resolved_input_path.parent.name
    )

    episode_records: list[LiberoEpisodeRecord] = []
    window_records: list[LiberoWindowRecord] = []
    task_counts: Counter[str] = Counter()
    num_frames = 0
    dropped_short_episodes = 0

    for source_file in files:
        task_id = _derive_task_id_from_hdf5_path(source_file)
        with h5py.File(source_file, "r") as handle:
            demo_names = sorted(handle["data"].keys(), key=_parse_demo_index)
            if max_demos_per_file is not None:
                demo_names = demo_names[:max_demos_per_file]
            for demo_name in demo_names:
                if max_windows is not None and len(window_records) >= max_windows:
                    break
                demo_group = handle["data"][demo_name]
                episode_record, arrays = _build_episode_record_and_arrays(
                    source_file=source_file,
                    suite_name=effective_suite_name,
                    task_id=task_id,
                    demo_name=demo_name,
                    demo_group=demo_group,
                    camera_keys=camera_keys,
                )
                episode_records.append(episode_record)
                task_counts[task_id] += 1
                num_frames += episode_record.num_frames
                episode_windows = _build_window_records(
                    episode=episode_record,
                    arrays=arrays,
                    context_len=context_len,
                    horizon=horizon,
                    stride=stride,
                )
                if not episode_windows:
                    dropped_short_episodes += 1
                for window in episode_windows:
                    if max_windows is not None and len(window_records) >= max_windows:
                        break
                    window_records.append(window)
            if max_windows is not None and len(window_records) >= max_windows:
                break

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = resolved_output_dir / "episodes.jsonl"
    windows_path = resolved_output_dir / "windows.jsonl"
    summary_path = resolved_output_dir / "summary.json"
    write_episode_jsonl(episode_records, episodes_path)
    write_window_jsonl(window_records, windows_path)

    warnings: list[str] = []
    if dropped_short_episodes:
        warnings.append("Some episodes were too short for the requested context/horizon.")
    if max_windows is not None and len(window_records) >= max_windows:
        warnings.append("Export stopped early because max_windows was reached.")
    summary = LiberoExportSummary(
        input_path=str(resolved_input_path),
        output_dir=str(resolved_output_dir),
        suite_name=effective_suite_name,
        context_len=context_len,
        horizon=horizon,
        stride=stride,
        num_files=len(files),
        num_episodes=len(episode_records),
        num_windows=len(window_records),
        num_frames=num_frames,
        dropped_short_episodes=dropped_short_episodes,
        tasks={str(key): int(value) for key, value in sorted(task_counts.items())},
        paths={
            "episodes_jsonl": str(episodes_path),
            "windows_jsonl": str(windows_path),
            "summary_json": str(summary_path),
        },
        warnings=warnings,
    )
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def export_libero_hdf5_suite_collection(
    input_root: str | Path,
    output_dir: str | Path,
    *,
    suite_names: Sequence[str] = DEFAULT_LIBERO_SUITE_NAMES,
    context_len: int = 2,
    horizon: int = 16,
    stride: int = 4,
    max_files_per_suite: int | None = None,
    max_demos_per_file: int | None = None,
    max_windows_per_suite: int | None = None,
    camera_keys: Sequence[str] = DEFAULT_CAMERA_KEYS,
) -> LiberoSuiteCollectionExportSummary:
    """Export multiple LIBERO suite directories and write combined JSONL files."""

    if not suite_names:
        raise LiberoHdf5ExportError("suite_names must be non-empty")
    normalized_suite_names = [str(name).strip() for name in suite_names]
    if any(not name for name in normalized_suite_names):
        raise LiberoHdf5ExportError("suite_names must not contain empty names")
    if len(set(normalized_suite_names)) != len(normalized_suite_names):
        raise LiberoHdf5ExportError("suite_names must be unique")

    resolved_input_root = Path(input_root).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    suite_summaries: dict[str, LiberoExportSummary] = {}
    task_counts: Counter[str] = Counter()
    warnings: list[str] = []
    episode_jsonl_paths: list[Path] = []
    window_jsonl_paths: list[Path] = []

    for suite_name in normalized_suite_names:
        suite_input_dir = resolved_input_root / suite_name
        if not suite_input_dir.exists():
            raise LiberoHdf5ExportError(f"missing LIBERO suite directory: {suite_input_dir}")
        suite_output_dir = resolved_output_dir / suite_name
        suite_summary = export_libero_hdf5_windows(
            input_path=suite_input_dir,
            output_dir=suite_output_dir,
            suite_name=suite_name,
            context_len=context_len,
            horizon=horizon,
            stride=stride,
            max_files=max_files_per_suite,
            max_demos_per_file=max_demos_per_file,
            max_windows=max_windows_per_suite,
            camera_keys=camera_keys,
        )
        suite_summaries[suite_name] = suite_summary
        for task_id, count in suite_summary.tasks.items():
            task_counts[f"{suite_name}/{task_id}"] += int(count)
        for warning in suite_summary.warnings:
            warnings.append(f"{suite_name}: {warning}")
        episode_jsonl_paths.append(Path(suite_summary.paths["episodes_jsonl"]))
        window_jsonl_paths.append(Path(suite_summary.paths["windows_jsonl"]))

    episodes_path = resolved_output_dir / "episodes.jsonl"
    windows_path = resolved_output_dir / "windows.jsonl"
    summary_path = resolved_output_dir / "summary.json"
    _concatenate_jsonl(episode_jsonl_paths, episodes_path)
    _concatenate_jsonl(window_jsonl_paths, windows_path)

    summary = LiberoSuiteCollectionExportSummary(
        input_root=str(resolved_input_root),
        output_dir=str(resolved_output_dir),
        suite_names=normalized_suite_names,
        context_len=context_len,
        horizon=horizon,
        stride=stride,
        num_suites=len(suite_summaries),
        num_files=sum(item.num_files for item in suite_summaries.values()),
        num_episodes=sum(item.num_episodes for item in suite_summaries.values()),
        num_windows=sum(item.num_windows for item in suite_summaries.values()),
        num_frames=sum(item.num_frames for item in suite_summaries.values()),
        dropped_short_episodes=sum(item.dropped_short_episodes for item in suite_summaries.values()),
        tasks={str(key): int(value) for key, value in sorted(task_counts.items())},
        suites={name: suite_summaries[name].to_dict() for name in normalized_suite_names},
        paths={
            "episodes_jsonl": str(episodes_path),
            "windows_jsonl": str(windows_path),
            "summary_json": str(summary_path),
        },
        warnings=warnings,
    )
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def write_episode_jsonl(records: Sequence[LiberoEpisodeRecord], path: str | Path) -> None:
    _write_jsonl(records, path)


def write_window_jsonl(records: Sequence[LiberoWindowRecord], path: str | Path) -> None:
    _write_jsonl(records, path)


def read_episode_jsonl(path: str | Path) -> list[LiberoEpisodeRecord]:
    return [LiberoEpisodeRecord.from_dict(payload) for payload in _read_jsonl(path)]


def read_window_jsonl(path: str | Path) -> list[LiberoWindowRecord]:
    return [LiberoWindowRecord.from_dict(payload) for payload in _read_jsonl(path)]


def _build_episode_record_and_arrays(
    *,
    source_file: Path,
    suite_name: str,
    task_id: str,
    demo_name: str,
    demo_group: Any,
    camera_keys: Sequence[str],
) -> tuple[LiberoEpisodeRecord, dict[str, list[list[float]]]]:
    obs_group = demo_group["obs"]
    actions = _rows_to_float_lists(demo_group["actions"][()])
    ee_states = _load_ee_states(obs_group)
    gripper_states = _rows_to_float_lists(obs_group["gripper_states"][()])
    joint_states = _rows_to_float_lists(obs_group["joint_states"][()])
    num_frames = len(actions)
    _ensure_aligned_lengths(
        {
            "actions": len(actions),
            "ee_states": len(ee_states),
            "gripper_states": len(gripper_states),
            "joint_states": len(joint_states),
        },
        source_file=source_file,
        demo_name=demo_name,
    )
    camera_refs = {}
    for camera_key in camera_keys:
        if camera_key not in obs_group:
            raise LiberoHdf5ExportError(f"{source_file}/{demo_name} is missing camera {camera_key}")
        dataset = obs_group[camera_key]
        if len(dataset.shape) < 4 or int(dataset.shape[0]) != num_frames:
            raise LiberoHdf5ExportError(
                f"{source_file}/{demo_name}/{camera_key} shape does not align with actions"
            )
        camera_refs[camera_key] = {
            "dataset": f"data/{demo_name}/obs/{camera_key}",
            "shape": [int(dim) for dim in dataset.shape],
            "dtype": str(dataset.dtype),
        }

    demo_index = _parse_demo_index(demo_name)[0]
    episode_id = f"{suite_name}__{task_id}__demo_{demo_index:03d}"
    episode = LiberoEpisodeRecord(
        schema_version=SCHEMA_VERSION,
        episode_id=episode_id,
        task_id=task_id,
        suite_name=suite_name,
        source_file=str(source_file),
        demo_name=demo_name,
        num_frames=num_frames,
        camera_refs=camera_refs,
        numeric_refs={
            "actions": f"data/{demo_name}/actions",
            "ee_states": f"data/{demo_name}/obs/ee_states"
            if "ee_states" in obs_group
            else f"data/{demo_name}/obs/ee_pos+ee_ori",
            "gripper_states": f"data/{demo_name}/obs/gripper_states",
            "joint_states": f"data/{demo_name}/obs/joint_states",
        },
        action_dim=len(actions[0]),
        ee_dim=len(ee_states[0]),
        gripper_dim=len(gripper_states[0]),
        joint_dim=len(joint_states[0]),
    )
    episode.validate()
    return episode, {
        "actions": actions,
        "ee_states": ee_states,
        "gripper_states": gripper_states,
        "joint_states": joint_states,
    }


def _build_window_records(
    *,
    episode: LiberoEpisodeRecord,
    arrays: dict[str, list[list[float]]],
    context_len: int,
    horizon: int,
    stride: int,
) -> list[LiberoWindowRecord]:
    num_frames = episode.num_frames
    first_anchor = context_len - 1
    last_anchor = num_frames - horizon - 1
    if last_anchor < first_anchor:
        return []

    windows: list[LiberoWindowRecord] = []
    for anchor_index in range(first_anchor, last_anchor + 1, stride):
        context_start = anchor_index - context_len + 1
        context_end = anchor_index + 1
        future_start = anchor_index + 1
        future_end = future_start + horizon
        action_start = anchor_index
        action_end = anchor_index + horizon
        anchor_ee_state = arrays["ee_states"][anchor_index]
        future_ee_states = arrays["ee_states"][future_start:future_end]
        action_chunk = arrays["actions"][action_start:action_end]
        window_id = (
            f"{episode.episode_id}__a{anchor_index:05d}"
            f"__c{context_start:05d}_{context_end:05d}"
            f"__f{future_start:05d}_{future_end:05d}"
        )
        window = LiberoWindowRecord(
            schema_version=SCHEMA_VERSION,
            window_id=window_id,
            episode_id=episode.episode_id,
            task_id=episode.task_id,
            suite_name=episode.suite_name,
            source_file=episode.source_file,
            demo_name=episode.demo_name,
            context_start=context_start,
            context_end=context_end,
            anchor_index=anchor_index,
            future_start=future_start,
            future_end=future_end,
            action_start=action_start,
            action_end=action_end,
            context_frame_indices=list(range(context_start, context_end)),
            future_frame_indices=list(range(future_start, future_end)),
            camera_keys=sorted(episode.camera_refs.keys()),
            anchor_ee_state=anchor_ee_state,
            future_ee_states=future_ee_states,
            future_delta_ee_states=_subtract_rows(future_ee_states, anchor_ee_state),
            action_chunk=action_chunk,
            current_gripper_state=arrays["gripper_states"][anchor_index],
            current_joint_state=arrays["joint_states"][anchor_index],
        )
        window.validate()
        windows.append(window)
    return windows


def _load_ee_states(obs_group: Any) -> list[list[float]]:
    if "ee_states" in obs_group:
        return _rows_to_float_lists(obs_group["ee_states"][()])
    if "ee_pos" not in obs_group or "ee_ori" not in obs_group:
        raise LiberoHdf5ExportError("obs group must contain ee_states or ee_pos + ee_ori")
    ee_pos = _rows_to_float_lists(obs_group["ee_pos"][()])
    ee_ori = _rows_to_float_lists(obs_group["ee_ori"][()])
    if len(ee_pos) != len(ee_ori):
        raise LiberoHdf5ExportError("ee_pos and ee_ori lengths do not match")
    return [pos + ori for pos, ori in zip(ee_pos, ee_ori)]


def _rows_to_float_lists(array: Any) -> list[list[float]]:
    rows = array.tolist()
    if not isinstance(rows, list):
        raise LiberoHdf5ExportError("expected array-like data")
    if rows and not isinstance(rows[0], list):
        rows = [[value] for value in rows]
    return [[float(value) for value in row] for row in rows]


def _subtract_rows(rows: list[list[float]], anchor: list[float]) -> list[list[float]]:
    return [[float(value - base) for value, base in zip(row, anchor)] for row in rows]


def _ensure_aligned_lengths(
    lengths: dict[str, int],
    *,
    source_file: Path,
    demo_name: str,
) -> None:
    if len(set(lengths.values())) != 1:
        raise LiberoHdf5ExportError(
            f"{source_file}/{demo_name} has inconsistent lengths: {lengths}"
        )


def _derive_task_id_from_hdf5_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        return stem[: -len("_demo")]
    return stem


def _parse_demo_index(demo_name: str) -> tuple[int, str]:
    try:
        return int(demo_name.split("_")[-1]), demo_name
    except ValueError:
        return 10**9, demo_name


def _validate_positive(value: int, field_name: str) -> None:
    if value <= 0:
        raise LiberoHdf5ExportError(f"{field_name} must be positive")


def _ensure_non_empty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LiberoHdf5ExportError(f"{field_name} must be a non-empty string")


def _ensure_float_list(values: list[float], field_name: str) -> None:
    if not values:
        raise LiberoHdf5ExportError(f"{field_name} must be non-empty")
    if not all(isinstance(value, (int, float)) for value in values):
        raise LiberoHdf5ExportError(f"{field_name} must contain only numeric values")


def _ensure_float_rows(rows: list[list[float]], field_name: str) -> None:
    if not rows:
        raise LiberoHdf5ExportError(f"{field_name} must be non-empty")
    for row in rows:
        _ensure_float_list(row, field_name)


def _write_jsonl(records: Sequence[Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _concatenate_jsonl(input_paths: Sequence[Path], output_path: str | Path) -> None:
    resolved_output_path = Path(output_path)
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("w", encoding="utf-8") as output_handle:
        for input_path in input_paths:
            with Path(input_path).open("r", encoding="utf-8") as input_handle:
                for line in input_handle:
                    if line.strip():
                        output_handle.write(line)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    payloads: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payloads.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise LiberoHdf5ExportError(f"invalid JSON at line {lineno}: {exc}") from exc
    return payloads
