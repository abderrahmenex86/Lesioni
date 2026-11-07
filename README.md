# Lesioni

Semantic segmentation of diseased regions on plant leaves, trained on PlantSeg.
It compares four architectures and keeps the training diagnostics that explain
which one generalises.

![Validation Dice comparison](docs/figs/01_model_comparison.png)

## What it does

- Trains four segmentation architectures behind one `--model_type` flag.
- Uses focal loss with per-class weighting, since healthy leaf background is
  about 80% of the pixels and dominates a plain cross-entropy objective.
- Optimises architecture and optimiser settings with Optuna, recording the
  search space so a run can be inspected afterwards.
- Writes checkpoints, metric history, and the hyperparameters of every run into
  `artifacts/<timestamp>_<model>_<classes>/`, and can resume from any of them.
- Plots Dice, IoU, and the generalisation gap across models from the stored
  histories.

## Results

The comparison plots are the record: validation Dice, validation IoU, and
generalisation gap per architecture are in `docs/figs/01_model_comparison.png`,
`06_multi_model_iou_comparison.png`, and `04_generalization_gap_comparison.png`.

No numeric table is published in this repository, so the figures are the source
of truth rather than a summary table.

![IoU comparison](docs/figs/06_multi_model_iou_comparison.png)

## Dataset

PlantSeg, with 116 disease classes. Pixel balance is roughly 80% background to
20% lesion, which is why the loss is weighted and why accuracy alone would be
misleading.

## Run it

```bash
git clone https://github.com/abderrahmenex86/Lesioni
cd Lesioni
pip install -r requirements.txt

python tools.py --mode download --root_dir dataset
python tools.py --mode verify --root_dir dataset

python main.py --mode test --model_type unet --num_classes 116 --img_size 256
python main.py --mode optimize --model_type unet --num_classes 116 --img_size 256
python main.py --mode train --model_type unet --num_classes 116 --img_size 256 \
  --optimizer_type AdamW --learning_rate 0.0003 --weight_decay 0.0001
python main.py --mode infer --image_path dataset/plantseg/images/val/sample_leaf.jpg \
  --run_dir artifacts/20260628_190322_unet_116class
```

Interrupted runs resume with `python main.py --mode train --resume --num_classes
116`.

## Limits

- PlantSeg only. Other crops, lighting, and camera hardware are untested.
- 116 classes with a heavily imbalanced pixel distribution; rare diseases have
  little training signal.
- Results are plotted rather than tabulated, so comparing two runs means reading
  the figures.

## Stack

Python · PyTorch · TorchMetrics · Optuna · Matplotlib

## Related

- [Flora](https://github.com/abderrahmenex86/Flora) for plant species
  identification
- [Entomi](https://github.com/abderrahmenex86/Entomi) for insect and pest
  classification
