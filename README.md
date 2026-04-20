# Self-pruning on CIFAR-10: 93.31% sparsity in the main MLP submission, plus a 95.04% ResNet18 sanity-check baseline

This repository contains my Tredence AI Engineering case study submission. The core submission is a self-pruning MLP for CIFAR-10 where every weight has a learnable sigmoid gate and training combines cross-entropy with an L1 sparsity penalty. I implemented the pruning layer from scratch, ran a four-lambda sweep, analyzed the accuracy-versus-sparsity trade-off, and wrote up the findings in [report.md](report.md).

The engineering addition that mattered most was lambda annealing. Instead of applying full sparsity pressure from the first update, the model trains normally for five epochs and then ramps the regularizer over the next ten. That makes the pruning behavior much more stable and gives the network time to learn useful structure before it starts deleting connections.

After the main case study was complete, I also trained a separate GPU-only `ResNet18` baseline to verify that the data pipeline and training setup were healthy. That run reached `95.04%` best test accuracy and `94.91%` final test accuracy on an `NVIDIA GeForce RTX 3050 Ti Laptop GPU`, which confirmed that the self-pruning MLP's lower ceiling was architectural rather than a training bug.

## What makes this submission different

The part I am most glad I added was lambda annealing. Most implementations apply full sparsity pressure from epoch 1, which kills weights before the network learns anything useful. The warmup-and-ramp schedule here gave the network time to stabilize first, and the effect shows up clearly in the epoch-by-epoch breakdown in the report.

The ResNet baseline was not part of the original spec. I added it after the MLP results came in to answer an obvious question: was 55% accuracy a pruning problem or an architecture problem? The answer was architecture. `ResNet18` reached `95.04%` best test accuracy on the same CIFAR-10 pipeline.

## Key results

### Self-pruning MLP

| Lambda | Peak test acc (%) | Final test acc (%) | Final sparsity (%) | Takeaway |
| --- | --- | --- | --- | --- |
| `0.0` | 55.47 | 53.59 | 0.00 | Baseline with visible overfitting |
| `1e-4` | 57.25 | 55.67 | 31.70 | Best pure accuracy result |
| `1e-3` | 57.09 | 55.80 | 93.31 | Best compression/accuracy trade-off |
| `1e-2` | 55.86 | 42.52 | 99.90 | Over-pruned and unstable |

### Follow-up baseline

| Model | Best test acc (%) | Final test acc (%) | Best epoch | Hardware | Takeaway |
| --- | --- | --- | --- | --- | --- |
| `ResNet18` | 95.04 | 94.91 | 89 | `RTX 3050 Ti Laptop GPU` | Confirms the MLP ceiling was architectural, not a broken training stack |

## Where to look

- [report.md](report.md): final write-up with experiment analysis and figures.
- [train.py](train.py): main CLI for the self-pruning experiments.
- [src/self_pruning/model.py](src/self_pruning/model.py): custom `PrunableLinear` and MLP definition.
- [src/self_pruning/engine.py](src/self_pruning/engine.py): training loop, evaluation loop, and sweep runner.
- [resnet.py](resnet.py): single-file GPU-only ResNet18/34 baseline trainer.
- [tests/test_model.py](tests/test_model.py): smoke tests for parameter registration, shape flow, and scheduling logic.

## Figures

### Layer-wise sparsity across the lambda sweep

![Layer-wise sparsity](assets/figures/self_pruning_layerwise_sparsity.png)

### ResNet18 training curves

![ResNet training curves](assets/figures/resnet_training_curves.png)

## Quick start

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run a shape check:

```bash
python train.py --shape-check
```

Run the full self-pruning sweep:

```bash
# all experiments use torch.manual_seed(42) so the sweep is reproducible
python train.py --default-sweep --epochs 20 --output-dir outputs/case_study
```

Run the ResNet18 baseline on GPU:

```bash
python resnet.py --model resnet18
```

## Repository notes

- `outputs/` is gitignored because it contains large local training artifacts and checkpoints.
- The figures used in the markdown are copied into `assets/figures/` so the GitHub repo renders cleanly.
- The self-pruning MLP remains the core submission; the ResNet baseline is included as a benchmark and sanity check.
