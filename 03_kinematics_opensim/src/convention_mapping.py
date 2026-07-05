"""convention_mapping.py

Kinatrax -> OpenSim 座標慣例對應工具箱 (convention-mapping toolkit)
===================================================================

本模組是 chapter ``03_kinematics_opensim`` 的智識核心 (intellectual
centrepiece) 的純 Python 實作。它 **刻意不依賴 OpenSim**，只用 ``numpy`` 與
``scipy``，因此可以離線、可單元測試地驗證「把外部 markerless 關節角度
(joint angles) 對應到 OpenSim generalized coordinates」這件事最容易出錯的部分。

為什麼需要它 (motivation)
--------------------------
研究者的 Kinatrax markerless 系統輸出的是「關節角度」而不是 raw marker
軌跡，而其 **segment 定義、joint coordinate system、Euler/Cardan 旋轉順序、
sign convention、reference/zero pose 與單位** 都是 proprietary、且沒有文件保證
與 OpenSim 一致。任何 Kinatrax -> OpenSim 的角度對應都必須被 **當成未知、逆向
工程並驗證**，而不是直接照抄 [@wu2005]。本工具箱把這個對應拆解成四個可獨立
檢查的成分：

1. ``name_map``   -- source coordinate 名稱 -> 目標 OpenSim coordinate 名稱
2. per-axis ``sign``      -- 軸方向 (axis direction) 差異，+1 / -1
3. per-axis ``offset_deg``-- reference / zero pose 差異 (常數位移，單位:度)
4. Euler/Cardan 序列轉換  -- 由 :class:`EulerTriplet` 描述 (seq_from -> seq_to)

驗證方式一律採用 **round-trip**：對已知 pose 施加對應、再逆轉，確認能回到原值
(see :func:`round_trip_error`)，特別要在接近奇異點 (singular configuration)
的姿態密集取樣。

Intrinsic 與 extrinsic Euler 序列 (讀懂 scipy 的關鍵)
-----------------------------------------------------
一個剛體姿態 (orientation) 的旋轉矩陣 :math:`R` 只有一個，但把它拆成三個角
(a triplet) 的方式不唯一，因為旋轉不可交換
(:math:`R_x R_y \\neq R_y R_x`)。``scipy.spatial.transform.Rotation`` 用序列
字串區分兩種慣例：

* **Intrinsic** (大寫，如 ``"YXY"``, ``"ZXY"``)：每一次旋轉都繞著 **上一次
  旋轉後、隨著物體移動的新軸** (body-fixed / moving axes)。ISB 的
  humerothoracic 序列即 intrinsic Y-X'-Y''。
* **Extrinsic** (小寫，如 ``"yxy"``, ``"zxy"``)：每一次旋轉都繞著 **固定不動
  的世界軸** (space-fixed axes)。

同一個 :math:`R`，``Rotation.from_euler("ZXY", ...).as_euler("YXY", ...)``
就是「先由某慣例的 triplet 造出共享的 :math:`R`，再以另一慣例把 :math:`R`
拆回」。這正是 :func:`euler_reorder` 做的事。

Y-X-Y gimbal lock 的正確位置 (重要更正)
----------------------------------------
ISB humerothoracic 採用 **proper / symmetric Euler** 序列 Y-X-Y
(:math:`R = R_y(\\alpha)\\,R_x(\\beta)\\,R_y(\\gamma)`)，其中 :math:`\\alpha`
= plane of elevation、:math:`\\beta` = elevation、:math:`\\gamma` = axial
rotation [@wu2005]。奇異點 (gimbal lock) 發生在 **中間角** :math:`\\beta`
= 0 度 (手臂垂放於身側 / arm at side) 或 180 度 (完全過頭 / overhead)，此時兩個
Y 軸重合、只有 :math:`\\alpha + \\gamma` 有定義，plane 與 axial 會互相交換、
跳動。**並不是** 在 ~90 度 elevation；在 ~90 度時 Y-X-Y 反而是良態
(well-conditioned) 的。棒球投球的手臂大多在高抬臂範圍，因此若原始資料掃過
arm-at-side，Y-X-Y 的 plane/axial 會不穩定 -- 這是資料相關的假警報，不是對應
程式的 bug。改用 Tait-Bryan 序列 (如 X-Z-Y) 可避開此不連續。

Grood-Suntay JCS 提醒
---------------------
臨床上常見的 Joint Coordinate System [@groodsuntay1983] 使用兩根 body-fixed
軸加一根 floating 軸，且 **一般為非正交 (non-orthogonal)**；其角度在數學上等價
於某個特定 Cardan/Euler 序列。因此若 Kinatrax 內部其實是 JCS 風格的角度，本工具
箱的「序列轉換 + sign + offset」仍是逆向工程的正確起點，但務必用 round-trip
在全 ROM 上驗證 (見 chapter notes.md Section C 的 SOP)。

單位慣例 (unit convention)
--------------------------
本模組的角度 dict **一律以「度」(degrees)** 為預設 (符合 vendor 報告與
OpenSim .mot 檔的 ``inDegrees=yes`` 慣例)。所有函式都有 ``degrees`` 參數；
``offset_deg`` 永遠以度儲存，在 ``degrees=False`` 時會自動轉為弧度。OpenSim
的 state 內部是弧度、states .sto 一律 ``inDegrees=no``，換算留給下游 notebook
(05b) 在寫檔邊界處理。

References: [@wu2005] [@groodsuntay1983] [@degroot2001] [@seth2019] [@buffi2015]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "EulerTriplet",
    "ConventionMap",
    "euler_reorder",
    "apply_map",
    "round_trip_error",
    "detect_sign_offset",
    "infer_convention",
]

_VALID_AXES = set("xyzXYZ")


# ---------------------------------------------------------------------------
# small angle helpers
# ---------------------------------------------------------------------------
def _wrap(x: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Wrap angle(s) to (-180, 180] deg (or (-pi, pi] rad).

    Used so that a genuine 359.9-deg-vs--0.1-deg *identity* is not reported as
    a 360-deg error. Euler triplets returned by scipy live in canonical ranges,
    so wrapping keeps round-trip comparisons honest.
    """
    x = np.asarray(x, dtype=float)
    half = 180.0 if degrees else np.pi
    full = 360.0 if degrees else 2.0 * np.pi
    return (x + half) % full - half


def _circular_mean(x: np.ndarray, degrees: bool = True) -> float:
    """Circular mean of angles (robust when residuals straddle +/-180 deg)."""
    x = np.asarray(x, dtype=float)
    a = np.deg2rad(x) if degrees else x
    m = np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
    return float(np.rad2deg(m) if degrees else m)


# ---------------------------------------------------------------------------
# (1) the mapping data structures
# ---------------------------------------------------------------------------
@dataclass
class EulerTriplet:
    """Describe the Euler/Cardan resequencing of one 3-DOF joint.

    A single rotation triplet is the 3-DOF generalisation of the per-axis
    ``seq_from`` / ``seq_to`` mismatch. The three ``source_coords`` are the
    vendor angle channels (in ``seq_from`` order); the three ``target_coords``
    are the OpenSim generalized coordinates (in ``seq_to`` order). Both
    sequences are scipy-style strings -- **UPPERCASE = intrinsic**
    (body-fixed axes), **lowercase = extrinsic** (space-fixed axes).

    Example (shoulder): ISB humerothoracic ``"YXY"`` -> an OpenSim-style
    Cardan ``"ZXY"`` whose coordinates are ``plane_elv``, ``shoulder_elv``,
    ``axial_rot`` (Holzbaur-family naming) [@holzbaur2005].

    Note the proper-Euler ``"YXY"`` singularity at the *middle* angle
    (elevation) = 0 / 180 deg -- see the module docstring [@wu2005].
    """

    source_coords: Tuple[str, str, str]
    target_coords: Tuple[str, str, str]
    seq_from: str
    seq_to: str

    def __post_init__(self) -> None:
        self.source_coords = tuple(self.source_coords)  # type: ignore[assignment]
        self.target_coords = tuple(self.target_coords)  # type: ignore[assignment]
        if len(self.source_coords) != 3 or len(self.target_coords) != 3:
            raise ValueError("EulerTriplet needs exactly 3 source and 3 target coords.")
        for seq in (self.seq_from, self.seq_to):
            if len(seq) != 3 or any(c not in _VALID_AXES for c in seq):
                raise ValueError(
                    f"Invalid Euler sequence {seq!r}: need 3 letters from x/y/z "
                    "(uppercase=intrinsic, lowercase=extrinsic)."
                )


@dataclass
class ConventionMap:
    """A full source(Kinatrax)->target(OpenSim) coordinate-convention map.

    Captures the five ingredients of a convention mismatch in one object:

    * ``name_map``   : ``{source_coord_name: target_coord_name}`` for the
      **1-DOF pass-through** coordinates (e.g. ``r_elbow_flex``). Coordinates
      that belong to an :class:`EulerTriplet` are handled by that triplet and
      need *not* appear here (if they do, they are ignored by
      :func:`apply_map`).
    * ``sign``       : ``{target_coord_name: +1.0|-1.0}`` axis-direction flip.
      Missing entries default to ``+1.0``.
    * ``offset_deg`` : ``{target_coord_name: float}`` constant zero-pose offset
      in **degrees**. Missing entries default to ``0.0``.
    * ``euler_triplets`` : list of :class:`EulerTriplet`, one per 3-DOF joint,
      each carrying its own ``seq_from`` / ``seq_to``.

    ``sign`` and ``offset_deg`` are applied as a final per-axis affine
    correction, *after* the rotation-matrix resequencing:

    .. math::  q_{target} = sign \\cdot q_{reordered} + offset .

    Conceptually the ``euler_reorder`` step captures the geometric (axis-order)
    part of the mismatch assuming both conventions describe the same underlying
    segment frames, while ``sign``/``offset`` absorb the residual axis-direction
    and reference-pose differences that a pure resequencing cannot. This
    factorisation is a pragmatic modelling choice; it is only trustworthy once
    it passes a full-ROM round-trip (:func:`round_trip_error`).
    """

    name_map: Dict[str, str] = field(default_factory=dict)
    sign: Dict[str, float] = field(default_factory=dict)
    offset_deg: Dict[str, float] = field(default_factory=dict)
    euler_triplets: List[EulerTriplet] = field(default_factory=list)

    def target_coords(self) -> List[str]:
        """All OpenSim target coordinate names produced by this map."""
        names: List[str] = list(self.name_map.values())
        for tri in self.euler_triplets:
            names.extend(tri.target_coords)
        return names

    def triplet_source_names(self) -> set:
        """Source names consumed by any triplet (skipped by the 1-DOF path)."""
        s: set = set()
        for tri in self.euler_triplets:
            s.update(tri.source_coords)
        return s


# ---------------------------------------------------------------------------
# (2) sequence conversion via the shared rotation matrix
# ---------------------------------------------------------------------------
def euler_reorder(
    angles: Sequence[float] | np.ndarray,
    seq_from: str,
    seq_to: str,
    degrees: bool = True,
) -> np.ndarray:
    """Re-express an Euler/Cardan angle triplet in a different sequence.

    Builds the single shared rotation matrix :math:`R` from ``angles`` under
    ``seq_from``, then decomposes the *same* :math:`R` under ``seq_to``. This
    is the geometrically correct way to convert between rotation conventions --
    a per-channel copy is wrong whenever the axis order differs.

    Parameters
    ----------
    angles : array-like, shape (3,) or (N, 3)
        The triplet(s), in ``seq_from`` order.
    seq_from, seq_to : str
        scipy sequence strings. **UPPERCASE letters = intrinsic** (body-fixed,
        moving axes, e.g. ISB ``"YXY"``); **lowercase = extrinsic** (space-fixed
        axes). Mixing cases within one string is not allowed by scipy.
    degrees : bool
        Interpret and return angles in degrees (default) or radians.

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
        The triplet(s) in ``seq_to`` order, in scipy's canonical ranges:
        for proper-Euler sequences (e.g. ``"YXY"``) the middle angle is in
        [0, 180] deg; for Tait-Bryan (e.g. ``"ZXY"``) the middle angle is in
        [-90, 90] deg; the outer angles are in [-180, 180] deg.

    Notes
    -----
    ``euler_reorder`` is its own inverse:
    ``euler_reorder(euler_reorder(x, A, B), B, A) == x`` up to (a) angle
    wrapping and (b) gimbal-lock degeneracy. Near a singular middle angle
    (Y-X-Y at elevation 0/180 deg) scipy emits a gimbal-lock warning and the
    outer/third angle split becomes arbitrary -- the *pose* is still exact, but
    the individual plane/axial numbers are not recoverable [@wu2005].
    """
    arr = np.asarray(angles, dtype=float)
    r = Rotation.from_euler(seq_from, arr, degrees=degrees)
    return r.as_euler(seq_to, degrees=degrees)


# ---------------------------------------------------------------------------
# (3) apply a ConventionMap to a stream of source angles
# ---------------------------------------------------------------------------
def apply_map(
    source_angles: Mapping[str, Sequence[float] | np.ndarray | float],
    cmap: ConventionMap,
    degrees: bool = True,
) -> Dict[str, np.ndarray]:
    """Map a dict of source (Kinatrax) angle channels to OpenSim coordinates.

    Pipeline per coordinate:

    * **3-DOF triplet** : gather the three source channels, resequence with
      :func:`euler_reorder` (``seq_from`` -> ``seq_to``), then apply
      ``sign`` and ``offset_deg`` to each target axis.
    * **1-DOF pass-through** : ``target = sign * source + offset``.

    Parameters
    ----------
    source_angles : mapping name -> scalar or 1-D array
        Vendor angle channels. Every name referenced by ``cmap`` must be
        present.
    cmap : ConventionMap
    degrees : bool
        Units of the angle values (``offset_deg`` is always in degrees and is
        converted internally when ``degrees=False``).

    Returns
    -------
    dict name -> numpy.ndarray
        OpenSim-coordinate angle series (arrays, even for scalar input).
    """
    triplet_sources = cmap.triplet_source_names()
    out: Dict[str, np.ndarray] = {}

    def _sign(name: str) -> float:
        return float(cmap.sign.get(name, 1.0))

    def _offset(name: str) -> float:
        o = float(cmap.offset_deg.get(name, 0.0))
        return o if degrees else np.deg2rad(o)

    # --- 3-DOF joints (Euler resequencing) ---
    for tri in cmap.euler_triplets:
        try:
            cols = [np.atleast_1d(np.asarray(source_angles[c], dtype=float))
                    for c in tri.source_coords]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"apply_map: source channel {exc.args[0]!r} required by triplet "
                f"{tri.source_coords} not found in source_angles."
            ) from exc
        stacked = np.column_stack(cols)                      # (N, 3)
        reordered = euler_reorder(stacked, tri.seq_from, tri.seq_to, degrees=degrees)
        reordered = np.atleast_2d(reordered)                 # keep (N, 3)
        for i, tname in enumerate(tri.target_coords):
            out[tname] = _sign(tname) * reordered[:, i] + _offset(tname)

    # --- 1-DOF pass-through coordinates ---
    for sname, tname in cmap.name_map.items():
        if sname in triplet_sources:
            continue  # already produced by a triplet; name_map entry is decorative
        if sname not in source_angles:
            raise KeyError(
                f"apply_map: source channel {sname!r} required by name_map not "
                "found in source_angles."
            )
        arr = np.atleast_1d(np.asarray(source_angles[sname], dtype=float))
        out[tname] = _sign(tname) * arr + _offset(tname)

    return out


# ---------------------------------------------------------------------------
# (4) round-trip validation
# ---------------------------------------------------------------------------
def round_trip_error(
    original: Mapping[str, Sequence[float] | np.ndarray | float],
    recovered: Mapping[str, Sequence[float] | np.ndarray | float],
    degrees: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Per-coordinate max and RMS error between an original and recovered dict.

    Errors are angle-wrapped to (-180, 180] deg so a 360-deg-equivalent
    identity reads as ~0. Only coordinates present in *both* dicts are scored.

    Returns
    -------
    dict
        ``{coord: {"max": float, "rms": float}, ..., "overall": {...}}``
        with values in the same unit as the inputs (degrees by default). The
        special ``"overall"`` key aggregates across every scored sample.
    """
    keys = [k for k in original if k in recovered]
    result: Dict[str, Dict[str, float]] = {}
    all_diffs: List[np.ndarray] = []
    for k in keys:
        o = np.atleast_1d(np.asarray(original[k], dtype=float))
        r = np.atleast_1d(np.asarray(recovered[k], dtype=float))
        if o.shape != r.shape:
            raise ValueError(
                f"round_trip_error: shape mismatch for {k!r}: {o.shape} vs {r.shape}"
            )
        d = _wrap(r - o, degrees)
        result[k] = {
            "max": float(np.max(np.abs(d))),
            "rms": float(np.sqrt(np.mean(d ** 2))),
        }
        all_diffs.append(d.ravel())
    if all_diffs:
        alld = np.concatenate(all_diffs)
        result["overall"] = {
            "max": float(np.max(np.abs(alld))),
            "rms": float(np.sqrt(np.mean(alld ** 2))),
        }
    return result


# ---------------------------------------------------------------------------
# (5) detection helpers: reverse-engineer sign & offset from known poses
# ---------------------------------------------------------------------------
def detect_sign_offset(
    source_vals: Sequence[float] | np.ndarray,
    target_vals: Sequence[float] | np.ndarray,
    degrees: bool = True,
    slope_tol: float = 1e-6,
) -> Tuple[float, float]:
    """Infer (sign, offset_deg) for ONE axis from matched reference values.

    Given the same coordinate observed in both conventions across one or more
    known reference poses, fit ``target = sign * source + offset`` where
    ``sign`` is snapped to +/-1 (a pure convention difference has unit gain;
    a fitted gain far from +/-1 signals a *sequence/axis* mismatch that
    :func:`detect_sign_offset` cannot repair -- resequence first).

    * With >= 2 poses spanning a range, the slope is estimated by least squares
      and snapped to its sign.
    * With a single pose (or a constant source), the sign is assumed +1 and
      only the offset is recovered (a single sample cannot separate sign from
      offset -- supply at least two distinct poses).

    ``offset`` is a circular mean, robust to residuals straddling +/-180 deg.
    Returns ``(sign, offset)`` with ``offset`` in the input unit.
    """
    s = np.atleast_1d(np.asarray(source_vals, dtype=float))
    t = np.atleast_1d(np.asarray(target_vals, dtype=float))
    if s.shape != t.shape:
        raise ValueError("detect_sign_offset: source and target must match in shape.")

    if s.size >= 2 and float(np.ptp(s)) > slope_tol:
        A = np.column_stack([s, np.ones_like(s)])
        slope, _intercept = np.linalg.lstsq(A, t, rcond=None)[0]
        sign = 1.0 if slope >= 0.0 else -1.0
    else:
        sign = 1.0  # underdetermined: cannot distinguish sign from a single pose

    offset = _circular_mean(t - sign * s, degrees=degrees)
    return sign, offset


def infer_convention(
    source_poses: Mapping[str, Sequence[float] | np.ndarray],
    target_poses: Mapping[str, Sequence[float] | np.ndarray],
    name_map: Mapping[str, str],
    euler_triplets: Sequence[EulerTriplet] | None = None,
    degrees: bool = True,
) -> ConventionMap:
    """Build a ConventionMap by inferring sign & offset from known poses.

    The rotation SEQUENCES (``seq_from`` / ``seq_to`` inside each supplied
    :class:`EulerTriplet`) must already be known or hypothesised -- they cannot
    be inferred from a handful of poses. Given those, this routine recovers the
    residual per-axis ``sign`` and ``offset_deg``:

    * For each triplet, the source channels are first resequenced to the target
      sequence, then :func:`detect_sign_offset` is fit per target axis against
      ``target_poses``.
    * For each 1-DOF entry in ``name_map`` not already consumed by a triplet,
      ``detect_sign_offset`` is fit directly.

    Parameters
    ----------
    source_poses, target_poses : mapping name -> 1-D array over reference poses
        The SAME set of known calibration poses expressed in both conventions.
        Use >= 2 well-separated poses (dense away from singular configurations).
    name_map : mapping source_name -> target_name
    euler_triplets : sequence of EulerTriplet, optional
        3-DOF joints whose sequence is known/hypothesised.

    Returns
    -------
    ConventionMap
        With ``name_map``, inferred ``sign``/``offset_deg``, and the supplied
        ``euler_triplets``. Always validate the result with a full-ROM
        :func:`round_trip_error`.
    """
    triplets = list(euler_triplets) if euler_triplets else []
    sign: Dict[str, float] = {}
    offset_deg: Dict[str, float] = {}
    consumed: set = set()

    for tri in triplets:
        consumed.update(tri.source_coords)
        cols = [np.atleast_1d(np.asarray(source_poses[c], dtype=float))
                for c in tri.source_coords]
        stacked = np.column_stack(cols)
        reordered = np.atleast_2d(
            euler_reorder(stacked, tri.seq_from, tri.seq_to, degrees=degrees)
        )
        for i, tname in enumerate(tri.target_coords):
            s_axis = reordered[:, i]
            t_axis = np.atleast_1d(np.asarray(target_poses[tname], dtype=float))
            sg, off = detect_sign_offset(s_axis, t_axis, degrees=degrees)
            sign[tname] = sg
            # offset stored in DEGREES by contract; convert back if working in rad
            offset_deg[tname] = off if degrees else float(np.rad2deg(off))

    for sname, tname in name_map.items():
        if sname in consumed:
            continue
        sg, off = detect_sign_offset(
            source_poses[sname], target_poses[tname], degrees=degrees
        )
        sign[tname] = sg
        offset_deg[tname] = off if degrees else float(np.rad2deg(off))

    return ConventionMap(
        name_map=dict(name_map),
        sign=sign,
        offset_deg=offset_deg,
        euler_triplets=triplets,
    )


# ---------------------------------------------------------------------------
# example / self-test: undo a deliberate convention mismatch
# ---------------------------------------------------------------------------
def _selftest() -> int:
    """Manufacture a known Kinatrax<->OpenSim mismatch and prove we can undo it.

    Because the ground truth is known, the round-trip error must be ~0. This
    doubles as the module's smoke test (run ``python convention_mapping.py``).
    """
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 72)
    print("convention_mapping self-test: undo a deliberate Kinatrax<->OpenSim "
          "mismatch")
    print("=" * 72)

    # --- ground truth OpenSim coordinates (degrees), smooth bounded motion ---
    fs = 200.0
    t = np.arange(0.0, 1.0, 1.0 / fs)
    bump = (1.0 - np.cos(2.0 * np.pi * 0.5 * t)) / 2.0   # 0 -> 1 -> 0-ish, smooth
    truth = {
        "plane_elv":     30.0 + 20.0 * bump,   # ~[30, 50] deg
        "shoulder_elv":  20.0 + 40.0 * bump,   # ~[20, 60] deg  -> middle axis of ZXY
        "axial_rot":    -15.0 + 25.0 * bump,   # ~[-15, 10] deg
        "r_elbow_flex":  10.0 + 90.0 * bump,   # ~[10, 100] deg (1-DOF)
    }

    # --- the TRUE convention differences we will pretend not to know ---
    # shoulder target sequence ZXY: axis order (plane=Z, shoulder_elv=X-middle,
    # axial=Y). Keeping shoulder_elv on the middle axis with sign +1 keeps the
    # pre-image inside ZXY's [-90, 90] canonical range so the round-trip is exact.
    SIGN = {"plane_elv": -1.0, "shoulder_elv": +1.0, "axial_rot": +1.0,
            "r_elbow_flex": -1.0}
    OFFSET = {"plane_elv": 0.0, "shoulder_elv": 0.0, "axial_rot": 30.0,
              "r_elbow_flex": 90.0}
    SEQ_TO, SEQ_FROM = "ZXY", "YXY"      # OpenSim Cardan  <-  ISB humerothoracic

    # --- manufacture the pseudo-Kinatrax source by INVERTING the map ---
    # forward map:  q_target = sign * reorder(source, from->to) + offset
    # so:           source   = reorder( (q_target - offset)/sign , to->from )
    pre = np.column_stack([
        (truth["plane_elv"]    - OFFSET["plane_elv"])    / SIGN["plane_elv"],
        (truth["shoulder_elv"] - OFFSET["shoulder_elv"]) / SIGN["shoulder_elv"],
        (truth["axial_rot"]    - OFFSET["axial_rot"])    / SIGN["axial_rot"],
    ])
    src_shoulder = np.atleast_2d(euler_reorder(pre, SEQ_TO, SEQ_FROM))
    source = {
        "KinaShoulderPlane": src_shoulder[:, 0],
        "KinaShoulderElev":  src_shoulder[:, 1],
        "KinaShoulderAxial": src_shoulder[:, 2],
        "KinaElbowFlex":     (truth["r_elbow_flex"] - OFFSET["r_elbow_flex"])
                             / SIGN["r_elbow_flex"],
    }

    # --- build the ConventionMap we would hand-author in the notebook ---
    shoulder_tri = EulerTriplet(
        source_coords=("KinaShoulderPlane", "KinaShoulderElev", "KinaShoulderAxial"),
        target_coords=("plane_elv", "shoulder_elv", "axial_rot"),
        seq_from=SEQ_FROM, seq_to=SEQ_TO,
    )
    cmap = ConventionMap(
        name_map={"KinaElbowFlex": "r_elbow_flex"},
        sign=dict(SIGN),
        offset_deg=dict(OFFSET),
        euler_triplets=[shoulder_tri],
    )

    # --- apply and validate ---
    recovered = apply_map(source, cmap)
    err = round_trip_error(truth, recovered)
    print("\nRound-trip error (deg) after undoing the mismatch:")
    for k in ("plane_elv", "shoulder_elv", "axial_rot", "r_elbow_flex", "overall"):
        print(f"  {k:14s}  max={err[k]['max']:.2e}  rms={err[k]['rms']:.2e}")
    passed = err["overall"]["max"] < 1e-6

    # --- Y-X-Y gimbal-lock demonstration (arm at side, NOT 90 deg) ---
    print("\nY-X-Y gimbal lock is at ELEVATION 0/180 deg (arm at side / overhead),")
    print("NOT at ~90 deg. Fixed plane=30, axial=15; sweep elevation across 0:")
    for e in (-2.0, -0.5, 0.0, 0.5, 2.0):
        R = Rotation.from_euler("YXY", [30.0, e, 15.0], degrees=True)
        rec = R.as_euler("YXY", degrees=True)
        print(f"  elev in={e:+5.2f} -> plane={rec[0]:+8.2f} elev={rec[1]:+6.2f} "
              f"axial={rec[2]:+8.2f}  (plane+axial={rec[0] + rec[2]:+8.2f} = stable)")
    print("  (at elev=90 the same Y-X-Y decomposition is well-conditioned.)")

    # --- detection demo: recover sign/offset from 3 calibration poses ---
    idx = np.array([0, len(t) // 2, len(t) - 1])
    sg_e, off_e = detect_sign_offset(source["KinaElbowFlex"][idx],
                                     truth["r_elbow_flex"][idx])
    stacked = np.column_stack([source["KinaShoulderPlane"][idx],
                               source["KinaShoulderElev"][idx],
                               source["KinaShoulderAxial"][idx]])
    reordered = np.atleast_2d(euler_reorder(stacked, SEQ_FROM, SEQ_TO))
    sg_a, off_a = detect_sign_offset(reordered[:, 2], truth["axial_rot"][idx])
    print("\nDetection from 3 calibration poses (sequence assumed known):")
    print(f"  r_elbow_flex : detected sign={sg_e:+.0f} offset={off_e:+.2f}  "
          f"(true sign={SIGN['r_elbow_flex']:+.0f} offset={OFFSET['r_elbow_flex']:+.2f})")
    print(f"  axial_rot    : detected sign={sg_a:+.0f} offset={off_a:+.2f}  "
          f"(true sign={SIGN['axial_rot']:+.0f} offset={OFFSET['axial_rot']:+.2f})")

    print("\n" + ("SELF-TEST PASSED" if passed else "SELF-TEST FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())
