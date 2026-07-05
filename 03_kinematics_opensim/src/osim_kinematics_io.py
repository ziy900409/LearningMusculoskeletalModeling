"""osim_kinematics_io.py

Kinematics I/O helpers for Chapter 03 (``03_kinematics_opensim``) of the
"從零到博士資格考" musculoskeletal-modeling repo.

The chapter's applied goal is an ANGLE-DRIVEN pipeline: external (markerless,
KinaTrax-style) joint ANGLES -> OpenSim generalized coordinates -> a
coordinates ``.mot`` / states ``.sto`` file -> Inverse Dynamics / analysis /
a MocoTrack reference. These utilities are the file-I/O and round-trip
verification primitives that pipeline needs, using the OpenSim **4.x** Python
API (``import opensim as osim``).

#1 GOTCHA, encoded throughout this module: RADIANS vs DEGREES.
  * Coordinate default values and ``getValue``/``setValue`` are in RADIANS.
  * A coordinates ``.mot`` may store DEGREES *only if* it carries the header
    metadata ``inDegrees=yes``.
  * A states ``.sto`` is ALWAYS in radians (``inDegrees=no``).
Set the header wrong (or omit it) and OpenSim silently interprets degrees as
radians -> nonsensical, huge motion.

Design notes
  * Functions are small and side-effect-light so they can be unit-tested with
    the bundled ``arm26`` model and a synthetic "pseudo-KinaTrax" angle stream.
  * ``times`` may be any 1-D sequence (list / np.ndarray). ``*_dict`` arguments
    map a column/coordinate name to a 1-D sequence of the same length.
  * Where an OpenSim 4.x call could not be verified verbatim against a specific
    installed version, a robust fallback and a ``TODO`` comment are provided.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

# --- Guarded OpenSim import ------------------------------------------------
# The pip/conda `opensim` package ships the API bindings only; the example
# .osim/.trc/.mot files come with the OpenSim GUI install or the
# opensim-org/opensim-models GitHub repo.
try:
    import opensim as osim
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise ImportError(
        "osim_kinematics_io requires the OpenSim 4.x Python package, which was "
        "not found. Install it into your conda environment, e.g.:\n"
        "    conda create -n osim -c opensim-org opensim python=3.11\n"
        "    conda activate osim\n"
        "(see the OpenSim scripting docs for platform-specific notes). "
        f"Original import error: {exc}"
    ) from exc


__all__ = [
    "check_opensim",
    "load_model",
    "list_coordinates",
    "coordinates_to_mot",
    "coordinate_states_dict",
    "states_to_sto",
    "read_mot",
    "mot_is_in_degrees",
    "prescribe_and_report",
]


# ---------------------------------------------------------------------------
# Small internal helpers
# ---------------------------------------------------------------------------
def _rotational(coord) -> bool:
    """True if a coordinate is angular (values in radians / degrees-in-file)."""
    try:
        return coord.getMotionType() == osim.Coordinate.Rotational
    except Exception:
        # getMotionType() emits deprecation-style warnings in 4.x because
        # MotionType moved from TransformAxis to Coordinate at 4.0. If the enum
        # comparison ever fails, assume rotational (all upper-limb joint angles
        # in this chapter are). TODO: confirm osim.Coordinate.Rotational exists
        # on the installed version.
        return True


def _motion_type_name(coord) -> str:
    """Return 'Rotational' / 'Translational' / 'Coupled' (best effort)."""
    try:
        mt = coord.getMotionType()
        for label in ("Rotational", "Translational", "Coupled"):
            if hasattr(osim.Coordinate, label) and mt == getattr(osim.Coordinate, label):
                return label
    except Exception:
        pass
    return "Unknown"


def _coord_abspath(coord) -> str:
    """Absolute component path, e.g. '/jointset/r_shoulder/r_shoulder_elev'."""
    try:
        return coord.getAbsolutePathString()
    except Exception:
        # Fallback for the standard layout <model>/jointset/<joint>/<coord>.
        # TODO: confirm getAbsolutePathString() on the installed OpenSim version.
        return "/jointset/%s/%s" % (coord.getJoint().getName(), coord.getName())


def _get_coordinate(coord_set, name):
    """Fetch a coordinate by name, with a helpful error listing what exists."""
    try:
        return coord_set.get(name)
    except Exception as exc:
        avail = [coord_set.get(i).getName() for i in range(coord_set.getSize())]
        raise ValueError(
            f"coordinate '{name}' not found in model. Available: {avail}"
        ) from exc


def _as_series(times, data_dict):
    """Coerce (times, {name: seq}) to (np.ndarray, {name: np.ndarray}) and
    validate that every series matches the number of time samples."""
    t = np.asarray(list(times), dtype=float)
    n = t.size
    series = {}
    for name, vals in data_dict.items():
        arr = np.asarray(list(vals), dtype=float)
        if arr.size != n:
            raise ValueError(
                f"length mismatch: column '{name}' has {arr.size} samples "
                f"but 'times' has {n}"
            )
        series[name] = arr
    return t, series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_opensim() -> str:
    """Return the installed OpenSim version string (``osim.GetVersion()``)."""
    return osim.GetVersion()


def load_model(path: str):
    """Load an ``.osim`` model and return it (NOT yet system-initialized).

    Call ``model.initSystem()`` before any state-dependent read
    (``getValue``, ``getTransformInGround``, ``equilibrateMuscles``, a
    ``Manager``). Range/default getters do NOT need a state.
    """
    model = osim.Model(str(path))
    model.finalizeConnections()  # resolve the component tree / absolute paths
    return model


def list_coordinates(model) -> list:
    """Inspect a model's coordinates without needing a realized state.

    Returns a list of dicts, one per coordinate, with keys:
      ``name``, ``path`` (absolute component path), ``value_state`` /
      ``speed_state`` (the ``.../value`` and ``.../speed`` state-variable
      paths), ``joint`` (owning joint), ``motion_type``
      ('Rotational'/'Translational'/'Coupled'), ``is_rotational`` (bool, i.e.
      degrees-applicable), ``default_value`` and ``default_speed`` (RADIANS for
      rotational coords), ``range_min`` / ``range_max``, and the default flags
      ``locked`` / ``clamped`` / ``prescribed``.

    The CoordinateSet is iterable in 4.x; index access is used here for
    maximum version robustness (``cs.getSize()`` / ``cs.get(i)``).
    """
    cs = model.getCoordinateSet()
    rows = []
    for i in range(cs.getSize()):
        coord = cs.get(i)
        abspath = _coord_abspath(coord)
        rows.append(
            {
                "name": coord.getName(),
                "path": abspath,
                "value_state": abspath + "/value",
                "speed_state": abspath + "/speed",
                "joint": coord.getJoint().getName(),
                "motion_type": _motion_type_name(coord),
                "is_rotational": _rotational(coord),
                "default_value": coord.getDefaultValue(),      # radians (rot.)
                "default_speed": coord.getDefaultSpeedValue(),
                "range_min": coord.getRangeMin(),
                "range_max": coord.getRangeMax(),
                "locked": coord.getDefaultLocked(),
                "clamped": coord.getDefaultClamped(),
                "prescribed": coord.getDefaultIsPrescribed(),
            }
        )
    return rows


def coordinates_to_mot(times, angles_dict, path, in_degrees=True):
    """Write a coordinates ``.mot`` (or ``.sto``) from angle time series.

    Column labels are BARE coordinate names (e.g. ``r_shoulder_elev``), which
    is what a coordinates ``.mot`` uses (contrast: a states ``.sto`` needs
    absolute ``/jointset/.../value`` paths).

    Values are written VERBATIM. ``in_degrees`` only sets the ``inDegrees``
    header flag, so it MUST match the units of the values you pass:
      * ``in_degrees=True``  -> values already in degrees, header ``yes``.
      * ``in_degrees=False`` -> values in radians,        header ``no``.
    Getting this wrong makes OpenSim read degrees as radians.

    Uses ``TimeSeriesTable`` + ``STOFileAdapter`` (the scalar/double adapter),
    which writes the ``nRows`` / ``nColumns`` / ``inDegrees`` header
    automatically and picks ``.mot`` vs ``.sto`` format from the extension.

    Returns the output path (str).
    """
    t, series = _as_series(times, angles_dict)
    labels = list(series.keys())

    table = osim.TimeSeriesTable()
    table.setColumnLabels(labels)
    for k in range(t.size):
        row = [float(series[name][k]) for name in labels]
        table.appendRow(float(t[k]), osim.RowVector(row))
    table.addTableMetaDataString("inDegrees", "yes" if in_degrees else "no")

    osim.STOFileAdapter.write(table, str(path))
    # --- Legacy fallback (OpenSim <=3.x, if STOFileAdapter were unavailable):
    #     storage = osim.Storage(); storage.setColumnLabels(<Frame,time,cols>);
    #     for each k: storage.append(t[k], osim.ArrayDouble([...vals...]));
    #     storage.setInDegrees(in_degrees); storage.print(str(path))
    return str(path)


def coordinate_states_dict(model, times, angles_dict, in_degrees=True,
                           finite_diff_speed=True):
    """Build a states dict (RADIANS) from a coordinate-angle dict.

    Resolves each coordinate's absolute path from ``model`` and emits a
    ``.../value`` channel (converted to radians for rotational coords) and,
    if ``finite_diff_speed``, a ``.../speed`` channel (``np.gradient`` of the
    radian values w.r.t. time). Hand the result to :func:`states_to_sto`, or
    to MocoTrack via ``TableProcessor``.
    """
    t, series = _as_series(times, angles_dict)
    cs = model.getCoordinateSet()
    states = {}
    for name, arr in series.items():
        coord = _get_coordinate(cs, name)
        rad = np.deg2rad(arr) if (in_degrees and _rotational(coord)) else arr.copy()
        base = _coord_abspath(coord)
        states[base + "/value"] = rad
        if finite_diff_speed:
            speed = np.gradient(rad, t) if t.size > 1 else np.zeros_like(rad)
            states[base + "/speed"] = speed
    return states


def states_to_sto(times, states_dict, path, model=None):
    """Write a full states ``.sto`` file.

    Column labels MUST be ABSOLUTE state-variable paths, e.g.
    ``/jointset/r_shoulder/r_shoulder_elev/value`` and ``.../speed``, or muscle
    states like ``/forceset/BIClong/activation``. Values are in RADIANS and the
    header is ALWAYS ``inDegrees=no`` (states files are never in degrees).

    If ``model`` is given, keys are validated against
    ``model.getStateVariableNames()`` and a warning is emitted for any label
    that is not a state variable of the model (OpenSim tools would reject it).

    Returns the output path (str).
    """
    t, series = _as_series(times, states_dict)

    if model is not None:
        svn = model.getStateVariableNames()  # ArrayStr
        valid = {svn.get(i) for i in range(svn.getSize())}
        unknown = [k for k in series if k not in valid]
        if unknown:
            warnings.warn(
                "states_to_sto: these labels are not state variables of the "
                f"model and OpenSim tools may reject the file: {unknown}"
            )

    labels = list(series.keys())
    table = osim.TimeSeriesTable()
    table.setColumnLabels(labels)
    for k in range(t.size):
        row = [float(series[name][k]) for name in labels]
        table.appendRow(float(t[k]), osim.RowVector(row))
    table.addTableMetaDataString("inDegrees", "no")  # states are always radians

    osim.STOFileAdapter.write(table, str(path))
    return str(path)


def read_mot(path):
    """Read a scalar ``.mot``/``.sto`` (coordinates or states) file.

    Returns
    -------
    times : np.ndarray
        The independent (time) column.
    columns : dict[str, np.ndarray]
        ``{column_label: values}`` for every dependent column.

    Values are returned VERBATIM as stored. To learn whether rotational values
    are in degrees, call :func:`mot_is_in_degrees` (uses
    ``Storage.isInDegrees()``).
    """
    table = osim.TimeSeriesTable(str(path))  # reads .sto/.mot scalar tables
    times = np.asarray(list(table.getIndependentColumn()), dtype=float)
    labels = list(table.getColumnLabels())
    n = table.getNumRows()

    columns = {}
    for label in labels:
        col = table.getDependentColumn(label)  # SimTK vector view
        try:
            vals = [col[i] for i in range(n)]
        except Exception:
            # Some builds expose element access as .get(i) rather than [i].
            vals = [col.get(i) for i in range(n)]
        columns[label] = np.asarray(vals, dtype=float)
    return times, columns


def mot_is_in_degrees(path) -> bool:
    """True if a ``.mot``/``.sto`` stores rotational values in DEGREES.

    Uses the legacy ``Storage`` reader's ``isInDegrees()`` (reads the header's
    ``inDegrees`` flag), which is handy for verifying files written elsewhere.
    """
    return bool(osim.Storage(str(path)).isInDegrees())


def prescribe_and_report(model, times, angles_dict, in_degrees=True):
    """Kinematically drive ``model`` through coordinate trajectories and read
    the realized coordinate values back (a round-trip check).

    Method (KINEMATICS ONLY, no dynamics/integration):
      1. ``state = model.initSystem()``.
      2. For each named coordinate, if locked, transiently unlock it
         (``setLocked(state, False)``) so ``setValue`` takes effect -- a locked
         coordinate is held fixed and ignored otherwise.
      3. For each time sample: ``setValue(state, radians)`` on every driven
         coordinate (the 2-arg form enforces constraints, i.e. calls
         ``assemble`` so any ``CoordinateCouplerConstraint`` such as a
         scapulohumeral rhythm is satisfied and dependent coords follow),
         then ``realizePosition`` and ``getValue`` to read back.

    Returns ``{coord_name: realized values}`` in the SAME units as the input
    (degrees if ``in_degrees``), so you can diff input vs output directly. For
    a clean mapping the max abs difference should be ~machine-epsilon, except
    for coordinates whose value is overridden by a constraint (e.g. a driven
    dependent scapula coordinate) or clipped by clamping ([rangeMin, rangeMax]).

    Notes / caveats
      * Coordinates flagged prescribed (``getDefaultIsPrescribed``) are governed
        by their prescribed function during ``assemble``; unset with
        ``setDefaultIsPrescribed(False)`` + re-``initSystem`` if you need
        ``setValue`` to win.
      * For SPEED on large models, replace the per-value assemble with a batched
        pass: ``coord.setValue(state, v, False)`` (enforceConstraints=False) for
        all coords, then a single ``model.assemble(state)``.

    Dynamic alternatives (NOT used here, mentioned per the chapter spec):
      * PRESCRIBED FORWARD SIM: ``coord.setPrescribedFunction(osim.Function)``,
        ``coord.setDefaultIsPrescribed(True)``, then integrate with the 4.x
        Manager: ``mgr = osim.Manager(model); state.setTime(t0);
        mgr.initialize(state); state = mgr.integrate(tFinal)``.
      * ``PositionMotion`` / ``StatesTrajectory``: build states and realize each.
      * A ``Kinematics`` analysis (``osim.Kinematics`` + ``AnalyzeTool`` over a
        coordinates ``.mot``) to dump ``*_Kinematics_q/u/dudt.sto``.
    """
    t, series = _as_series(times, angles_dict)
    state = model.initSystem()
    cs = model.getCoordinateSet()

    driven = []
    for name in series:
        coord = _get_coordinate(cs, name)
        if coord.getLocked(state):
            coord.setLocked(state, False)  # transient unlock so setValue applies
        driven.append((name, coord))

    realized = {name: np.empty(t.size) for name, _ in driven}
    for k in range(t.size):
        state.setTime(float(t[k]))
        for name, coord in driven:
            v = float(series[name][k])
            if in_degrees and _rotational(coord):
                v = math.radians(v)
            coord.setValue(state, v)  # 2-arg: enforces constraints (assemble)
        model.realizePosition(state)
        for name, coord in driven:
            rv = coord.getValue(state)
            if in_degrees and _rotational(coord):
                rv = math.degrees(rv)
            realized[name][k] = rv
    return realized
