# Multi-Camera 3D Crop Studio

A cross-platform Windows and Ubuntu/Linux desktop application for capturing or importing RGB-D data, colored point clouds,
and triangle meshes. It provides standard translation/rotation gizmos, an interactive 3D
crop box, a movable export coordinate frame, visible surface normals, and consistent PLY,
PCD, and STL output.

## Features

- Direct Intel RealSense streaming with synchronized color and aligned depth
- One-shot Photoneo capture through PhoXi Control and its optional Python API
- Native Photoneo PRAW/PMRAW File Camera input when the Photoneo API is installed
- Generic NPZ, PLY, PCD, PTX, XYZ, TXT, CSV, STL, OBJ, OFF, VTK, and VTP import
- Optional companion RGB image attachment for matching point topologies
- Live RGB, depth, and interactive 3D point-cloud views
- A crop transform gizmo with RGB translation arrows, RGB rotation rings, and six
  dedicated face-sphere handles for resizing
- Red preview of points currently inside the crop box
- Configurable export coordinate frame with translation, rotation, and automatic extrema
- Selectable normal estimation and direction with visible 3D arrow glyphs
- Binary PLY and PCD export with per-point RGB color and optional normal vectors
- Organized RGB-D meshing, imported triangle-topology preservation, and approximate
  surface reconstruction for unorganized clouds
- RGB PNG, 16-bit depth PNG, selection preview, and camera metadata export
- Built-in Demo Mode for testing the complete workflow without a camera
- Automated geometry and file-format tests
- Reversible point-cloud enhancement with editable GUI parameters
- Reference-plane flattening, millimetre height colors, and display-only vertical scaling

## Requirements

- 64-bit Windows 10/11 or a current 64-bit Ubuntu release
- 64-bit Python 3.10, 3.11, or 3.12
- An Intel RealSense depth camera is optional for RealSense live mode
- A Photoneo sensor and PhoXi Control 1.17 or later are optional for Photoneo live/PRAW mode
- A USB 3 connection and the vendor runtime are required for the corresponding camera mode

## Quick installation on Windows

1. Install 64-bit Python 3.10, 3.11, or 3.12 and keep the Python Launcher option enabled.
2. Double-click `install_windows.bat`. It creates `.venv` and installs all dependencies.
3. When installation finishes, run `run_windows.bat`.

Manual installation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

## Quick installation on Ubuntu

Install Python and the common Qt/OpenGL runtime libraries, clone this repository, and run
the included installer:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libgl1 libegl1 \
  libxkbcommon-x11-0 libxcb-cursor0 libxcb-xinerama0
git clone https://github.com/YOUR_GITHUB_ACCOUNT/realsense-point-cloud-studio.git
cd realsense-point-cloud-studio
chmod +x install_ubuntu.sh run_ubuntu.sh
./install_ubuntu.sh
./run_ubuntu.sh
```

Replace `YOUR_GITHUB_ACCOUNT` with the repository owner shown in the final GitHub URL.
The scripts keep all Python packages inside `.venv`; they do not modify the system Python.

For Intel RealSense live capture, connect the camera over USB 3. If Linux device access is
denied, install the official librealsense udev rules/runtime for your Ubuntu version. Demo
Mode and imported files do not require camera hardware or librealsense system packages.

## Usage

1. Choose an input:
   - Click **Demo mode** for an immediate hardware-free test.
   - Click **Connect RealSense** for a live Intel RealSense stream.
   - Enter a Photoneo device ID, or leave it empty when exactly one device is visible,
     then click **Capture Photoneo**.
   - Click **Open 3D / RGB-D file** for an offline file.
2. For a generic file, choose **Auto-detect**, **Metres**, or **Millimetres** before opening
   it. PTX and native Photoneo inputs are always interpreted as millimetres.
3. If a point cloud has no embedded color, click **Attach RGB** and choose an image whose
   pixel count or resolution matches the point topology.
4. Position the object and click **Freeze frame** for a stable final selection.
5. Use the three manipulation buttons directly below **Freeze frame**. Choose **Crop:
   translate / rotate** to use the
   red, green, and blue arrows for X/Y/Z and the matching rings for RX/RY/RZ. Choose **Crop:
   resize** to activate the six yellow face-resizing spheres. Only one mouse
   tool is active at once, preventing VTK picker conflicts. Selected points are red.
6. Choose the output root folder and maximum mesh-edge length. Values from 10 to 30 mm
   work well for many tabletop scenes. Lower values prevent triangles from crossing
   large depth discontinuities.
7. Click **Confirm and save PLY + PCD + STL**. Every capture is written to a new
   timestamped folder, so existing exports are never overwritten.

## Export coordinate frame

The **Export coordinate origin** panel controls both the location and orientation of the
coordinate frame used by exported PLY, PCD, and STL files. All three formats receive the
same rigid transform, so they remain aligned when loaded together. Available origin modes
are:

- Camera origin `(0, 0, 0)`
- Manual origin, movable with the RGB translation arrows or X/Y/Z fields in millimetres
- Center of the selected point cloud's axis-aligned bounding box
- Minimum and maximum X points
- Minimum and maximum Y points (`+Y` points down, so Minimum Y is the visual top)
- Minimum and maximum Z points

Automatic extrema modes use the complete XYZ coordinate of the point with the selected
minimum or maximum axis value. Changing any X/Y/Z field after choosing an automatic mode
switches the application to Manual mode for fine adjustment. Dragging an automatic origin
also switches to Manual. Red, green, and blue arrows translate the frame along X, Y, and Z;
the corresponding rings rotate RX, RY, and RZ. Rotation can also be entered numerically in
degrees. The original source-space origin, all three export axes, and the applied transform
are recorded in `metadata.json`.

Selecting **Manual**, editing an origin or rotation field, or pressing **Origin move /
rotate** automatically activates the export-frame gizmo. Press **Crop resize** at the top
of the panel to restore the six yellow face handles at any time.

## Surface normals

Enable **Show normal direction arrows** to draw yellow arrows at the selected points. The
panel provides these controls:

- **Automatic** uses imported triangle topology when available, organized cross-products
  for RGB-D images, and PCA neighbourhood estimation for unorganized point clouds.
- **Outward from object** produces true outside-facing normals for a closed imported mesh,
  including rings and other objects with inner surfaces. A closed mesh is the only input
  that defines inside versus outside unambiguously.
- **Toward current 3D view** is normally the correct choice for a single-view RealSense or
  Photoneo scan because only the camera-visible surface was measured. **Away current view**
  selects the opposite side.
- **Consistent estimated direction** retains the estimator's connected orientation, while
  **Flip estimated direction** reverses every active normal.
- **PCA neighbours**, **Arrow length**, and **Maximum arrows** control estimation and display
  density. **Use selected crop only** limits the glyphs and direction anchor to the crop.

When **Write normals to PLY and PCD** is enabled, PLY receives `nx`, `ny`, and `nz`
properties and PCD receives `normal_x`, `normal_y`, and `normal_z` fields. STL stores
triangle face normals as part of the STL format. Normal method and orientation are written
to `metadata.json`.

## Point-cloud enhancement controls

The application always keeps the latest raw frame separate from the enhanced frame. Click
**Reset to raw** to discard every active enhancement and reconstruct the view from the
untouched input. Click **10 mm preset** for a conservative starting point intended for
small stationary objects on a dominant reference surface, then adjust any parameter and
click **Apply**.

- **Multi-frame median** combines recent live-camera depth frames. More frames reject more
  random noise but require the camera and object to remain stationary. It has no benefit
  for a single imported frame.
- **Temporal smoothing** applies an exponential live-stream filter. A lower alpha is
  smoother but responds more slowly to movement.
- **Hole filling** fills only small invalid regions that are strongly surrounded by valid
  pixels. Radius and pass count control how aggressively gaps are filled.
- **Spatial bilateral** smooths depth while limiting smoothing across depth edges. Kernel
  size, depth sigma in millimetres, and strength are independently editable.
- **Statistical outliers** removes isolated 3D points using a neighbourhood and standard
  deviation threshold. It is disabled in the 10 mm preset because aggressive values can
  remove valid edges of very small objects.
- **Flatten reference plane** estimates the dominant plane robustly and transforms it to
  Z=0. The tolerance controls which points contribute to the final plane estimate.
- **Height color map** colors each point by signed height above the reference plane. The
  minimum, maximum, and RGB blending values are editable.
- **Vertical exaggeration** enlarges height in the 3D view only. PLY, PCD, STL, numeric
  origin fields, and metadata keep the real, non-exaggerated dimensions.
- **3D point size** changes only the renderer. **Top view** looks perpendicular to the
  flattened plane. **Show red selection preview** can be disabled when the red overlay
  obscures the height colors.

### Empty-table normalization and small-object detection

For a fixed camera looking at a physically flat table, empty-table normalization is more
effective than aggressive smoothing because it cancels repeatable local stereo-depth
corrugation without flattening real objects.

1. Remove every object from the table and keep the camera rigidly fixed.
2. Resume the live stream, set the reference frame count (60 is a good starting value),
   and click **Capture empty table**.
3. Wait until the reference status reports completion. The application stores the median
   depth at every pixel and the average RGB image.
4. Place the objects without touching or moving the camera. Click **10 mm preset**, wait
   for the multi-frame history to stabilize, and freeze the stream.
5. The reference is aligned to Z=0. Repeatable per-pixel table error is subtracted, points
   inside **Snap measured floor to zero** are set exactly to zero, and connected regions
   above **Detect raised objects** are marked green.
6. Enable **RGB change assist** to mark image regions that changed from the empty table.
   RGB-only candidates are outlined in magenta; this is useful for black materials that
   return invalid depth, but it does not invent missing 3D height.

The minimum component size suppresses isolated noise. A 4 mm height threshold, 12-pixel
minimum component, and 2.5 mm floor band are conservative starting values for 10 mm
objects. The status line reports the robust floor-noise sigma. If that value approaches the
object height, improve the physical capture before lowering the detection threshold.

Use **Save reference** and **Load reference** to reuse a calibrated empty table. The camera,
resolution, focus, and table position must remain unchanged. Capture a new reference after
any camera movement. **Clear empty-table reference** returns to ordinary plane processing.

The enhanced geometry visible in the application is used for crop and export, except for
display-only vertical exaggeration and height coloring. The complete enhancement settings
are written to `metadata.json` so an export can be reproduced.

## Supported inputs

| Input | Color | Topology | Notes |
|---|---:|---:|---|
| Intel RealSense live | Yes | Organized | Aligned RGB-D through `pyrealsense2` |
| Photoneo live | Texture or color image when supplied | Organized | Requires PhoXi Control and `phoxi_api` |
| `.praw`, `.pmraw` | Yes when recorded | Organized | Attached as a PhoXi Control File Camera |
| `.npz` | Yes | Organized | Lossless application RGB-D archive |
| `.ptx` | Intensity or RGB | Organized | Photoneo coordinates are converted from mm to m |
| `.pcd` | Optional | Organized or unorganized | ASCII and binary PCD; compressed binary is not supported |
| `.ply` | Optional | Points or triangles | ASCII/binary reading is provided by VTK |
| `.xyz`, `.txt`, `.csv` | Optional RGB columns | Unorganized | X Y Z, optionally R G B |
| `.stl`, `.obj`, `.off`, `.vtk`, `.vtp` | Format-dependent | Triangle mesh | Existing faces are retained when cropped |

Auto unit detection treats a maximum axis span above 10 as millimetres and smaller scenes
as metres. This is a convenience heuristic, not format metadata. Select the explicit unit
when the physical scale is known. STL itself has no standard stored unit.

### Photoneo setup

Install PhoXi Control 1.17 or later with its optional Python API, start PhoXi Control, and
make sure the camera is visible there. The application imports `phoxi_api` only when a
Photoneo action is used, so all other modes work without that package. For direct capture,
use **Capture Photoneo**. For `.praw` or `.pmraw`, use **Open 3D / RGB-D file**; the file is
temporarily attached as a PhoXi Control File Camera.

On Windows, the Python wrapper locates `PhoXi_API.dll` through the `PHOXI_CONTROL_PATH`
environment variable, which normally points to
`C:\Program Files\Photoneo\PhoXiControl-<version>` and is configured by the PhoXi Control
installer. On Ubuntu, install the vendor-provided PhoXi Control and Python API for Linux
and ensure its shared libraries are visible to the application environment. When multiple
versions are installed, verify that the Python package and native runtime versions match.

If the Python API is unavailable, open the native recording in PhoXi Control and export an
organized PLY or PTX file. Import that file here. PRAW/PMRAW are proprietary native
recordings and cannot be decoded reliably as ordinary PLY-style files.

## Export layout

```text
outputs/capture_YYYYMMDD_HHMMSS_mmm/
|-- selected_cloud.ply       # Colored point cloud with optional normals; metres
|-- selected_cloud.pcd       # Colored point cloud with optional normals; metres
|-- selected_mesh.stl        # Surface mesh; numeric coordinates use millimetres
|-- rgb.png                  # Full RGB frame
|-- depth_mm_16bit.png       # Raw 16-bit depth image in millimetres
|-- selection_preview.png    # Selected RGB pixels on a black background
`-- metadata.json            # Intrinsics, crop planes, export origin, counts, and settings
```

STL does not store an explicit unit. This application writes STL vertex coordinates in
millimetres for compatibility with common CAD applications and slicers. PLY and PCD
coordinates remain in metres. The STL represents only surfaces visible to the RGB-D
camera or reconstructed/imported surface, so it may be open at crop or occlusion boundaries
and is not guaranteed to be watertight. Surface reconstruction from an unorganized point
cloud is approximate; the exported PLY and PCD still retain every selected source point.

## 3D controls

- Orbit: left-drag on empty space
- Pan: `Shift + left-drag`
- Zoom: mouse wheel
- First choose one of the three manipulation buttons below **Freeze frame**
- Crop translation/rotation mode: drag a red, green, or blue arrow or ring
- Crop resize mode: drag any yellow face sphere; the opposite face remains fixed
- Export-frame mode: drag its RGB translation arrows or rotation rings
- Freeze or resume: Space
- Export: `Ctrl+S`

## Troubleshooting

- If the camera cannot be opened, close RealSense Viewer and any other application using
  the device, verify the USB 3 cable and port, and try the 640 x 480 profile.
- If `pyrealsense2` cannot be installed, use a supported 64-bit Python build.
- If 3D rendering is slow, increase **Display stride**. This does not reduce export quality.
- If long triangles bridge the object and background, reduce **Maximum mesh edge**.
- If PRAW/PMRAW cannot be opened, verify that PhoXi Control is running and its optional
  `phoxi_api` package is available to the same Python environment.
- If an imported model appears 1,000 times too large or too small, reopen it with the
  explicit **Metres** or **Millimetres** setting.

## Tests

Run the hardware-independent test suite with either PowerShell or a Bash-compatible shell:

```text
python -m unittest discover -v
```

The included visual smoke-test script launches Demo Mode, captures the real Qt/VTK window,
exports a cropped sample, and saves RGB and point-cloud preview images:

```text
python tools/visual_smoke_test.py
```
