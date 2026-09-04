# Three-dimensional heat world model experiments

## Scope

- Specimen: C45 cuboid, 60 x 40 x 20 mm.
- Base grid: 9 x 7 x 5 nodes, 315 temperature states.
- Refined grid: 17 x 13 x 9 nodes, 1,989 temperature states.
- Boundary: uniform convection and gray-body radiation on all six faces.
- Material: temperature-dependent C45 conductivity and heat capacity.
- Horizon: 300 transitions at 1 s per step.
- Dataset: 96 BDF trajectories, split into 56 train, 12 validation, 12 ID test, and 16 control-OOD test trajectories.

## Learned dynamics

The model is a controlled residual 3D convolutional network. Inputs are the current field, furnace command, four thermal parameters, and three coordinate channels. Both comparisons use 16 hidden channels, three residual blocks, three-step closed-loop training, and seed 42. The checkpoint is selected by validation-set 300-step rollout RMSE.

The data-only model uses zero physics weight. The physics-constrained model uses a semi-implicit six-face finite-volume residual with weight 0.1. The stronger three-dimensional weight was used because the numerical scale of this residual differs from the one-dimensional experiment.

## Base-grid commands

```bash
PYTHONPATH=src uv run --no-sync python scripts/train_three_dimensional_world_model.py \
  --output-dir outputs/c45_three_dimensional \
  --trajectories 96 --steps 300 --epochs 40 --batch-size 256 \
  --only data_only --regenerate --device mps

PYTHONPATH=src uv run --no-sync python scripts/train_three_dimensional_world_model.py \
  --output-dir outputs/c45_three_dimensional \
  --trajectories 96 --steps 300 --epochs 25 --batch-size 256 \
  --physics-weight 0.1 --only physics_constrained --device mps

PYTHONPATH=src uv run --no-sync python scripts/analyze_three_dimensional_world_model.py
```

The formal run trained the data-only model for 40 epochs. The physics-weight study first evaluated 0.001 for 40 epochs, then evaluated 0.1 for 25 epochs. The final stored physics model is the 0.1 candidate. All reported test metrics are computed after validation-based checkpoint selection.

## Refined-grid run

The refined dataset was regenerated from the same random seed and the same 96 sampled operating conditions. Shape-compatible convolution tensors were initialized from the corresponding base-grid checkpoint. Coordinate buffers, normalization statistics, and every grid-dependent quantity were recomputed. Both candidates were trained for 12 epochs with batch size 64 and evaluated every two epochs. Their selected checkpoints were both recorded at epoch 6.

```bash
PYTHONPATH=src uv run --no-sync python scripts/train_three_dimensional_world_model.py \
  --output-dir outputs/c45_three_dimensional_17x13x9 \
  --trajectories 96 --steps 300 --shape 17 13 9 \
  --epochs 12 --batch-size 64 --evaluate-every 2 \
  --physics-weight 0.1 \
  --warm-start-dir outputs/c45_three_dimensional \
  --regenerate --device mps

PYTHONPATH=src uv run --no-sync python scripts/analyze_three_dimensional_world_model.py \
  --output-dir outputs/c45_three_dimensional_17x13x9 \
  --figure-split id_test --reference-shape 25 19 13
```

The refined-grid validation rollout RMSE was 4.724 degC for the data-only model and 4.335 degC for the physics-constrained model. ID-test rollout RMSE was 3.983 and 4.082 degC, respectively. Relative to the base grid, these two ID values decreased by 10.8% and 11.9%. The ID implicit energy-residual RMSE decreased from 4.758 to 3.259 degC when the physics loss was used.

Control-OOD rollout RMSE on the refined grid was 17.014 degC for the data-only model and 18.104 degC for the physics-constrained model. The physics candidate improved 8 of 16 trajectories. Its paired mean difference was +0.653 degC with a 95% trajectory-bootstrap interval of [-0.346, 1.851] degC. No hyperparameter was selected from this held-out result.

The paper figure uses the ID-test trajectory whose data-only rollout RMSE is nearest the 75th percentile. At 192 s, a 25 x 19 x 13 BDF reference is compared with the 17 x 13 x 9 explicit Euler and learned fields. The snapshot RMSE values are 0.344, 5.150, and 4.530 degC for explicit Euler, data-only, and physics-constrained fields. Display interpolation is not used for the learned fields.

## Evidence boundary

This is a fixed-geometry, single-seed numerical extension. It demonstrates that the controlled recursive formulation can operate on a genuine three-dimensional field with 1,989 temperature states. It does not establish cross-geometry generalization, cross-grid inference, or real-furnace validity. The paired 95% interval for refined-grid control-OOD trajectory RMSE includes zero. The refined run therefore supports spatial-resolution feasibility and an ID energy-consistency result, not a universal OOD accuracy claim.

## Frozen artifact hashes

```text
dataset   e8ce5d8c25c9e528161618a046ab3a2f72f09bdd1855235e98a5277d37ad723e
data      02a8adb52357ca5be2d000035f8fd61cf1dfed9c3454d2edac10d90f42b2b9c9
physics   797e7634cc89566752d6a118735f699712489eb198b493f259ed6d13895f5900
metrics   18b76ac0681ee30a1a678afb0fac3ffaf6eed134a600200d86e83d0913b4af1c
figure    22a141c9fb4421424b1127d8d5ac35c383698a229ade7e48ef697c9313635660
pdf       bb2845142ea6076cd6fcbd7c43f3fdb2d40f6d582bf03bdc267e486e9ae71b93
```
