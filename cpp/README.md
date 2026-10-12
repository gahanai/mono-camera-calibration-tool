# C++ Calibration Tool

Standalone CMake-based C++ implementation of the calibration workflow described in the
[repository README](../README.md): a Qt5 + OpenCV desktop app that takes a monocular camera from
raw feed to a saved intrinsics + ground-plane homography, ready for a downstream BEV pipeline.

## Executables

- **`calibration_tool_qt`** — the actively developed tool. A single window with three tabs:
  - **Intrinsics**: camera + resolution selection (probes actual supported modes over MJPG so
    cameras aren't silently capped at a low default resolution), live preview with chessboard
    detection and ROS-style X/Y/Size/Skew coverage tracking, capture, and
    `cv::calibrateCamera` + save to YAML.
  - **Homography**: loads a saved intrinsics YAML and rectifies the live preview with it, then
    solves a ground-plane homography from 4 points — clicked manually, or auto-detected via 4
    ArUco markers. Camera position and height are recovered automatically with `cv::solvePnP`
    from the plane's known real-world size (width/height fields) and the 4 point
    correspondences, so no manual distance measurements to each corner are needed. A zoom loupe
    (press `Z` on the preview, click inside the magnified inset) helps place corner points
    precisely. A drag-a-bounding-box validation mode reports estimated distance-to-camera for
    sanity-checking against a tape measurement.
  - **Radar-Camera**: see [Radar-Camera Tab](#radar-camera-tab) below.
- **`bev_runner`** — loads a saved intrinsics + homography YAML pair, opens the camera at the
  intrinsics' resolution, and per frame: undistorts, then warps straight to a top-down BEV
  canvas by composing the saved homography with a meters-to-pixels transform. The view window
  auto-sizes to always include both the calibrated plane and the camera's own recovered ground
  position, with a 1m reference grid, the plane's footprint outline, and a camera-position
  marker drawn on top. Displays "Rectified" and "BEV" windows live (`q`/Esc to quit); see
  `./build/bev_runner --help` for resolution/margin flags.
- `intrinsics_calibrator_qt`, `extrinsics_calibrator_qt` — early standalone-window stubs
  superseded by the unified `calibration_tool_qt`. Not part of the CMake build.

Shared, Qt-free YAML readers for both executables live in
[`include/bev/calibration_io.hpp`](include/bev/calibration_io.hpp), so `calibration_tool_qt` and
`bev_runner` can never disagree about the file format.

## Default Asset Location

Calibration outputs default to `${HOME}/.calibration/openadas/cam<id>/`, where `<id>` is the
camera's device index (e.g. `/dev/video2` → `cam2`). This keeps multiple cameras' assets from
overwriting each other, and the path updates live as you switch the Camera dropdown:

```
${HOME}/.calibration/openadas/cam<id>/intrinsics.yaml
${HOME}/.calibration/openadas/cam<id>/intrinsics_images/   # saved calibration frames
${HOME}/.calibration/openadas/cam<id>/homography.yaml
${HOME}/.calibration/openadas/cam<id>/radar_cam_config.yaml
${HOME}/.calibration/openadas/cam<id>/radar_cam_captures/  # capture_0001/, capture_0002/, ...
${HOME}/.calibration/openadas/cam<id>/radar_extrinsics.yaml
```

Every path field has a Browse button if you want a different location instead.

## Radar-Camera Tab

Solves the 6-DOF rigid transform between a 4D radar and the camera using a trihedral corner
reflector rigidly bolted to a checkerboard, observed across 30-50 static captures at varied
range/azimuth/height. Implements the full pipeline from `radcam_calibplan.md` at the repo root:
closed-form Kabsch/Umeyama with mandatory determinant correction, RANSAC (exhaustive triple
enumeration at this capture-count scale, not random sampling), iterative gate tightening with an
inlier-drop guardrail, Ceres nonlinear refinement (Mode A: refine `(R,t)`; Mode B: jointly
estimate the reflector offset `X_B` too, since a bad caliper measurement of `X_B` is the one
systematic error the residuals otherwise can't detect), and two independent validation axes
(held-out residuals, and a ground-plane cross-check via the Homography tab's own saved YAML).

- **Config** (`radar_cam_config.yaml`): every threshold from the spec — gate radii per iteration,
  Doppler/RCS/persistence/spread thresholds, RANSAC and gate-iteration parameters, refinement
  mode, validation thresholds. Loaded via yaml-cpp; missing keys keep sensible defaults.
- **Capture** (`radar_cam_captures/capture_NNNN/`): `image.png` (rectified camera frame),
  `radar_dwell.npy` (a real, `numpy.load()`-compatible `(F, Dmax, 7)` float64 array — every raw
  per-frame detection, not just the aggregated result, per the spec's "log everything raw"
  requirement), optional `radar_bg.npy`, and `meta.yaml`.
  - Capture requires Camera Preview and Radar Live both running; it accumulates ~25 radar dwells
    (gated on the radar's own dwell `frame_id`, not raw CAN traffic) plus the current camera
    frame, runs board detection + PnP, and writes the directory if the capture passes the
    quality gates (full board detection, PnP reprojection RMS, board tilt).
- **Inspect** runs the pre-solve capture-distribution diagnostic (§9.3): warns if the reflector
  position cloud is too close to coplanar to constrain all 6 DOF, and reports range/azimuth/height
  spread (height is flagged specifically — it's the one operators forget to vary).
- **Solve** runs the full pipeline and writes `radar_extrinsics.yaml` (R, t, Euler angles, inlier
  count, RMS residuals, `X_B` estimated-vs-measured drift, per-stage clutter-filter funnel counts).
- **Validate** reports held-out 3D/pixel residual statistics, and — if a Homography-tab YAML is
  loaded — the ground-plane cross-check (derived radar mounting height/pitch/roll, compared
  against the tape-measure/inclinometer values in the config). This is a read-only diagnostic:
  it never feeds back into the solve.

### 2D radars (DRWIG LRR Pro)

Set `radar_type: drwig_lrr_pro` in the config to calibrate a DRWIG LRR Pro instead of the default
Columbus FC 4D (`radar_type: columbus_fc4d`). This radar is classic CAN and reports only 2D
tracked objects (x/y position + velocity; no elevation, no RCS), so the tab switches to:

- **Decode** (`include/bev/radarcam/drwig_decoder.hpp`): the 40 tracked-object slots of one radar,
  selected by `drwig_radar_id` (0..5). Every CAN ID a radar sends is shifted by a fixed per-radarId
  offset, so several radars can share one bus (e.g. rear-left = 4 → 0x550-0x5B0, rear-right = 5 →
  0x730-0x790). Radar frame: x forward, **y to the radar's right** (the DBC's raw sign), z down —
  write `coarse_euler_deg` for that frame. An optional `can_interface` key in the config sets the
  CAN Interface field on load.
- **Clutter filter**: gates in the radar plane (x/y only, since the reflector's height is lost),
  skips the RCS gate, and picks the detection closest to the prediction instead of the brightest.
- **Solve** (`include/bev/radarcam/planar_solve.hpp`): solves only yaw + in-plane offset — the DOF a
  2D radar can observe — on top of the config's coarse extrinsic, whose pitch, roll and mounting
  height are kept as measured. Ceres refinement is skipped (it would let those unobservable DOF
  drift). Solve/Validate residuals are in-plane distances, and the reflector may be at any height.

### Testing without radar hardware

The live CAN-FD reader can be exercised against a virtual CAN interface (`vcan0`), no radar
required — see the [Tests](#tests) section below for the one-time setup.

## Tests

Catch2-based unit tests for the Radar-Camera tab's algorithmic core
(`include/bev/radarcam/*.hpp`) live in `cpp/tests/`, and build/run independently of Qt:

```bash
cd cpp
cmake --build build -j
ctest --test-dir build --output-on-failure
```

One test (`SocketCanReader decodes a live dwell over a real AF_CAN socket`) needs a virtual CAN
interface and otherwise skips itself with setup instructions in its own failure message:

```bash
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
```

Diagnostic-plot renderers write preview PNGs (for manual visual review, since rendering isn't
numerically testable) to `${HOME}/.cache/openadas_radarcam_previews/`.

## Dependency Contract

- OpenCV 4.10.0 or higher
- CMake 3.10 or higher
- Qt5 Widgets
- C++17
- CUDA Toolkit (enabled by default)
- cuDNN (enabled by default)
- yaml-cpp — Radar-Camera tab's config/output YAML (`apt install libyaml-cpp-dev`)
- Ceres Solver — Radar-Camera tab's nonlinear refinement (`apt install libceres-dev`; pulls in
  Eigen, SuiteSparse, glog, gflags)
- Catch2 3 — unit test suite (`apt install catch2`), only needed if `BUILD_TESTS=ON` (default)
- Linux SocketCAN (kernel headers only, no extra package) — Radar-Camera tab's live CAN-FD reader

## OpenCV Resolution

The CMake project prefers OpenCV from `/opt/opencv/4.12.0` by default. An OpenCV build script is
provided in `cpp/assets/build_opencv.sh`.

- Preferred prefix is controlled by `BEV_OPENCV_PREFIX`.
- The build checks `${BEV_OPENCV_PREFIX}/lib/pkgconfig` first.
- If not found, it falls back to system pkg-config OpenCV packages.

Example using a custom prefix:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEV_OPENCV_PREFIX=/opt/opencv/4.12.0
```

## CUDA and cuDNN Defaults

CUDA and cuDNN are enabled by default in `CMakeLists.txt`. Neither is used by the calibration
tool itself yet — they're checked/linked ahead of the planned `bev_runner` GPU work.

- `ENABLE_CUDA=ON` (default)
- `ENABLE_CUDNN=ON` (default)

Disable explicitly if needed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_CUDNN=OFF
```

## Build

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Example build with explicit defaults shown:

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEV_OPENCV_PREFIX=/opt/opencv/4.12.0 -DENABLE_CUDA=ON -DENABLE_CUDNN=ON
cmake --build build -j
```

## Run

```bash
./build/calibration_tool_qt
./build/bev_runner [camera_index] [--intrinsics PATH] [--homography PATH] [--margin-m N] [--px-per-m N]
```

`bev_runner` defaults `camera_index` to `0` and both YAML paths to
`${HOME}/.calibration/openadas/cam<camera_index>/{intrinsics,homography}.yaml` — the same
default location the GUI tool saves to, so a plain `./build/bev_runner` picks up whatever you
just calibrated for camera 0.

## Typical Workflow

1. **Intrinsics tab**: pick the camera, scan/select its working resolution, set the chessboard
   inner-corner counts and square size, start preview, capture chessboard frames until coverage
   looks reasonable, then Calibrate + Save. This writes `cam<id>/intrinsics.yaml` and the
   captured frames.
2. **Homography tab**: pick the same camera, load that `intrinsics.yaml` (auto-filled by default),
   enter the ground-plane target's real-world width/height, start preview, and pick the 4 target
   corners (manual click or ArUco auto-detect) in TL → TR → BR → BL order. Use the zoom loupe
   (`Z`) for precise clicks. Solve + Save writes `cam<id>/homography.yaml`, including the
   solved homography, the `solvePnP`-recovered camera pose, and the intrinsics it was solved
   against.
3. Optionally use Validation mode to drag a box around a real object and compare the tool's
   estimated distance against a tape measurement.
4. Run `./build/bev_runner` (or `bev_runner <camera_index>` for a different camera) to see the
   live undistorted feed and its BEV warp side by side, using the same two YAML files.
5. **Radar-Camera tab** (needs a 4D radar on CAN-FD): fill in a `radar_cam_config.yaml` (board
   geometry, reflector offset `X_B`, coarse extrinsic, clutter/solver thresholds — see
   [Radar-Camera Tab](#radar-camera-tab)), start Camera Preview and Radar Live, then Capture the
   rig at 30-50 combinations of range (~5-40m), azimuth (full radar FOV), and height (>=3 distinct
   levels — the one operators forget). Run Inspect to sanity-check the capture spread before
   solving, then Solve, then Validate.
