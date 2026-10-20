# ResNet-18 on CIFAR-100 (from scratch)

A custom ResNet-18 implemented in PyTorch and trained from scratch on CIFAR-100, reaching **more than 79% test accuracy**.

Everything lives in a single notebook: `resnet18.ipynb`.

## Results

| Metric | Value |
|---|---|
| Dataset | CIFAR-100 (50k train / 10k test) |
| Model | ResNet-18 (CIFAR-style stem), trained from scratch |
| Epochs | 100 |
| Test accuracy | **> 79%** |

Experiments were tracked with Weights & Biases (project: `RESNET18`).

## What was changed to get here

### 1. Optimizer: Adam replaced by SGD with momentum and Nesterov
Adam was replaced with SGD (`momentum=0.9`, `nesterov=True`, `weight_decay=5e-4`). SGD with momentum generalizes better than Adam on CIFAR-style image classification.

```python
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4, nesterov=True)
```

### 2. One-cycle learning rate scheduler
A single `OneCycleLR` schedule is used for the whole run (warm-up to `max_lr=0.1`, then smooth decay to near zero). It is stepped **once per batch**, right after `optimizer.step()`.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.1, epochs=100, steps_per_epoch=len(trainLoader))
```

### 3. Mixed precision with `torch.amp.GradScaler`
Automatic mixed precision (`torch.autocast` + `GradScaler`) speeds up training and lowers GPU memory use. This matters because the CIFAR stem (see step 6) keeps feature maps large in the early layers.

```python
scaler = torch.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

### 4. CutMix with label smoothing, applied after the usual transforms
Standard augmentation (random crop with padding 4 and horizontal flip) is applied first. CutMix (`alpha=1.0`, `prob=0.5`) with `label_smoothing=0.1` is then applied on each batch (via `timm.data.Mixup`) to reduce overfitting. The loss is computed against the resulting **soft labels** using `nn.CrossEntropyLoss`.

```python
mixup_fn = Mixup(mixup_alpha=0.0, cutmix_alpha=1.0, prob=0.5,
                 label_smoothing=0.1, num_classes=100)
```

Note: training accuracy is measured on CutMix images against the original labels, so it reads lower than the model's true training accuracy.

### 5. Test accuracy above 79%
With the changes in this list, the model reaches more than 79% top-1 accuracy on the CIFAR-100 test set.

### 6. CIFAR-style stem instead of the ImageNet ResNet-18 stem
The classical ResNet-18 (designed for ImageNet) uses a 7x7 stride-2 convolution followed by max-pooling. That shrinks a 32x32 image to 8x8 before the first residual block, throwing away detail. This version changes the first block:

| | ImageNet ResNet-18 | This project |
|---|---|---|
| `conv1` | 7x7, stride 2 | **3x3, stride 1** |
| `maxpool` | 3x3, stride 2 | **removed** |
| Feature map sizes | 8 → 8 → 4 → 2 → 1 | **32 → 16 → 8 → 4** |

```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
self.bn1 = nn.BatchNorm2d(64)
# no maxpool
```

## Training setup

| Setting | Value |
|---|---|
| Batch size | 128 |
| Epochs | 100 |
| Peak learning rate | 0.1 (OneCycleLR) |
| Weight decay | 5e-4 |
| Augmentation | RandomCrop(32, pad=4), RandomHorizontalFlip, CutMix |
| Normalization | CIFAR-100 mean `(0.5071, 0.4865, 0.4409)`, std `(0.2673, 0.2564, 0.2762)` |
| Precision | Mixed (fp16 autocast) |

## Running it

```bash
pip install torch torchvision timm matplotlib numpy tqdm wandb
jupyter notebook resnet18.ipynb
```

The CIFAR-100 dataset downloads automatically to `./data` on first run. A CUDA GPU is recommended.

To log to Weights & Biases, run `wandb login` and set your own entity and project in `wandb.init(...)`.

## Repository structure

```
.
├── resnet18.ipynb   # model, training loop, evaluation, plots
├── README.md
└── .gitignore
```
