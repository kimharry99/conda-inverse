# conda-inverse

Reconstruct planar groups of cylindrical rebars from instance masks and
calibrated cameras. The Python distribution, import package and conda
environment retain their original names: `rebar-inverse`, `rebar_inverse`
and `rebar-inverse` respectively.

## Public source scope

This repository is a standalone source snapshot with a new Git history.
It contains the reconstruction library, camera and geometry utilities,
ground-truth evaluation, a dataset adapter, generic data-split utilities,
and their synthetic unit and integration tests.

Private datasets, project-specific experiment drivers, internal reports,
generated artifacts, model weights and original development history are
not distributed. No external dataset is needed for the CPU tests below.
The dataset CLI requires compatible data supplied by the caller.

## Installation

Use Python 3.11. The environment specification pins the original dependency
versions, including PyTorch 2.10.0 with CUDA 12.8, Mitsuba 3.9.1 and Dr.Jit 1.5.0.

```bash
conda env create -f environment.yml
conda activate rebar-inverse
python -m pip install --no-deps -e .
```

Alternatively, install into an existing compatible Python environment with
`python -m pip install -e .`. Reconstruction optimization needs an NVIDIA GPU,
a compatible CUDA driver and Mitsuba's `cuda_ad_rgb` variant. Geometry
initialization and the CPU test subset do not need GPU computation.

## Synthetic CPU example

```python
import numpy as np
from rebar_inverse import (
    Camera, Observation, RebarReconstructor, ReconstructionConfig,
)

mask = np.zeros((64, 64), dtype=np.int32)
mask[12:15, 5:59] = 1
mask[38:41, 5:59] = 2
K = np.array([[70., 0., 32.], [0., 70., 32.], [0., 0., 1.]])
camera = Camera(K, np.eye(4), width=64, height=64)
observation = Observation(mask, camera, name="synthetic/frame_0000")
api = RebarReconstructor(ReconstructionConfig(device="cpu"))
state = api.initialize([observation])
print(state.model.geometry().detach().numpy())
```

Masks are integer arrays: zero is background and positive values identify
individual rods. Intrinsics use pixel units; poses are **camera-to-world**
4×4 SE(3) matrices. Camera axes are +X right, +Y down and +Z forward.
Pixel centers are `(column + 0.5, row + 0.5)`. For OpenCV integer-center
intrinsics, add 0.5 to cx and cy. Correct distortion and skew before use.

The geometry array has one row per rod:
`[start_x, start_y, start_z, end_x, end_y, end_z, radius]`.

## CUDA reconstruction

For the same observations, initialize an API configured for CUDA:

```python
api = RebarReconstructor(ReconstructionConfig(
    device="cuda", max_sizes=(128, 256, 512),
    iterations=(30, 30, 40), num_planes=2,
    learning_rate=0.001, spp=64, seed=7,
))
result = api.reconstruct([observation])
result.save("outputs/run01")
```

Both `initialize` and `reconstruct` accept `mesh_ids`, an optional dictionary
mapping every observed instance ID to its known plane group. Automatic grouping
is a mask-width heuristic and does not guarantee physical layer membership.
Across multiple views, the caller must supply consistent instance IDs and
calibrated camera poses. Fully hidden rods are not inferred.

Every rod's center axis lies in its group's common plane. Positive lengths and
radii are optimized with PyTorch Adam. The renderer uses analytic cylinders
and disk endcaps with Mitsuba projective silhouette derivatives. By default,
directions are shared within each inferred direction family; orthogonality
between families is not imposed. Set `share_directions=False` to optimize
individual rod directions.

Outputs include `result.json` with geometry, configuration, metrics and
provenance, plus `predictions.npz` with initial and final masks. Coordinates
are in an arbitrary scale gauge, not recovered physical units. Mean coverage
IoU measures mask agreement, not 3D metric accuracy.

## Dataset adapter and command line

`V2Dataset(root)` reads a caller-provided directory of `scene/frame` folders.
The default local directory is `data/rebar/v2_2layer`. Each frame needs:

- `instance_id.npy`: a two-dimensional integer instance mask.
- `meta.json`: `camera.width`, `camera.height` and a normalized 3×3
  `intrinsics` matrix. The adapter multiplies its first row by image width
  and its second row by image height.
- Optional `semantic.png`: foreground must agree with the instance mask.
- `gt_cylinders.json` when requesting GT-derived plane groups or evaluation.
  Cylinder records contain `instance_id`, `mid`, `dir`, `length` and `radius`.
  GT geometry is interpreted in camera coordinates, in meters.

Create a manifest containing one dataset-relative `scene/frame` path per line.
A reconstruction manifest must refer to one scene. Multiple frames require
an explicit `--poses` JSON mapping every manifest entry to a camera-to-world
matrix; the adapter does not infer camera roll from look-at metadata.

```bash
python -m rebar_inverse \
  --dataset-root data/rebar/v2_2layer \
  --manifest my_manifest.txt --output outputs/run01 \
  --mesh-label-source mask --planes 2
```

The installed `rebar-reconstruct` entry point exposes the same command.
The default `--mesh-label-source gt` derives discrete plane membership from
GT; it does not copy GT geometry into initialization. Use `mask` explicitly
when GT is unavailable, or pass a `--mesh-ids` JSON dictionary. Inputs and
outputs must be kept in separate directories.

For GT evaluation:

```bash
python -m rebar_inverse.evaluate \
  --dataset-root data/rebar/v2_2layer \
  --result outputs/run01/result.json --output outputs/evaluation
```

Evaluation fits one positive uniform scale per mesh about the camera origin.
It reports transverse axis error, line distance, angle and diameter errors.
This is GT-assisted post-hoc alignment, not automatic metric reconstruction.
The Python `MeshScaleEvaluator` also accepts cylinder lists directly.

## Tests

Run the CPU subset with the public checkout first on the import path:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" python -m unittest \
  discover -s tests/unit/rebar_inverse -v
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$PWD" python -m unittest \
  tests.examples.test_cli tests.examples.test_evaluation \
  tests.examples.test_gt_labels -v
```

These cover camera projection, plane constraints, initialization, multiview
geometry, evaluation, data splitting, file I/O and CLI help using synthetic
inputs. The remaining integration tests require CUDA:

```bash
PYTHONPATH="$PWD" python -m unittest discover -s tests/examples -v
```

For this public snapshot, CPU checks were run; GPU rendering, optimization
and a fresh environment installation were not revalidated. No CI workflow
is configured.

## License

The source snapshot does not include an explicit license grant. No license
has been assigned as part of publication. Dependencies retain their own
licenses.
