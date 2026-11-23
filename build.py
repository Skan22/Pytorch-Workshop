"""Builds two notebooks from one source:
   photo_mixup_STUDENT.ipynb   (TODOs blanked out)
   photo_mixup_SOLUTIONS.ipynb (everything filled in)
"""
import nbformat as nbf

CELLS = []  # (kind, solution_source, student_source)


def md(text):
    CELLS.append(("md", text.strip("\n"), text.strip("\n")))


def code(sol, student=None):
    sol = sol.strip("\n")
    CELLS.append(("code", sol, (student if student is not None else sol).strip("\n")))


# --------------------------------------------------------------------------
md("""
# 🕵️ The Great Photo Mix-Up
### A PyTorch code-along

**The story.** Someone got into the family photo album and *rotated random photos* by 90°, 180° or 270°.
Nobody remembers which ones. Checking 60,000 photos by hand is not happening.

**Your mission.** Teach a CNN to look at a photo and say *"this one is rotated by X degrees"* — then use it to
automatically put the whole album back together.

**Why this is a nice problem:** we don't need any labels from humans! Every upright photo can be rotated by us,
so we *create* the labels for free (this idea is called *self-supervised learning*).

**What you'll practice:** tensors · custom `Dataset` · `DataLoader` · `nn.Module` · the training loop · evaluation · using a model.

> 🔧 Look for **`# TODO`** — that's where you type. Each TODO is followed by a ✅ **check cell**; if it passes, move on.
> Stuck? Ask me, or peek at the hint in the comments.

**Runtime → Change runtime type → GPU** (T4) if you're in Colab. It works on CPU too, just slower.
""")

md("## Part 0 · Setup")
code("""
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| running on:", device)

torch.manual_seed(0); random.seed(0); np.random.seed(0)
""")

md("""
## Part 1 · Meet the album (and warm up with tensors)

We'll use **CIFAR-10**: 60,000 tiny 32×32 colour photos (planes, cats, ships…). Everything in it starts out *upright*.
""")
code("""
train_raw = datasets.CIFAR10("./data", train=True,  download=True)
test_raw  = datasets.CIFAR10("./data", train=False, download=True)

CLASS_NAMES = train_raw.classes
# raw arrays are (N, 32, 32, 3) uint8 numpy -> turn into torch tensors
train_imgs = torch.from_numpy(train_raw.data)
test_imgs  = torch.from_numpy(test_raw.data)
print(train_imgs.shape, train_imgs.dtype)
""")

md("""
**Warm-up TODO #1 — tensor shapes.** PyTorch conv layers want images as `(N, C, H, W)` with floats in `[0, 1]`.
Our data is `(N, H, W, C)` with integers in `[0, 255]`. Convert `train_imgs` into the right format.
""")
code(
    """
train_x = train_imgs.permute(0, 3, 1, 2).float() / 255
test_x  = test_imgs.permute(0, 3, 1, 2).float() / 255
""",
    """
# Hint: .permute(...) reorders dimensions, .float() changes dtype, then divide.
train_x = ...  # TODO  -> shape (50000, 3, 32, 32), floats in [0, 1]
test_x  = ...  # TODO  -> shape (10000, 3, 32, 32)
""",
)
md("✅ **Check**")
code("""
assert train_x.shape == (50000, 3, 32, 32), f"wrong shape: {train_x.shape}"
assert train_x.dtype == torch.float32
assert 0.0 <= train_x.min() and train_x.max() <= 1.0
print("Tensor shapes ✅")
""")

code("""
def show(imgs, titles=None, ncols=8, size=1.6):
    \"\"\"imgs: tensor (N, 3, H, W) in [0,1]\"\"\"
    n = len(imgs)
    if n == 0:
        print("(nothing to show)"); return
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size * 1.15))
    for ax in np.array(axes).flatten(): ax.axis("off")
    for i in range(n):
        ax = np.array(axes).flatten()[i]
        ax.imshow(imgs[i].permute(1, 2, 0).cpu().clamp(0, 1))
        if titles is not None: ax.set_title(titles[i], fontsize=9)
    plt.tight_layout(); plt.show()

show(train_x[:16], [CLASS_NAMES[i] for i in train_raw.targets[:16]])
""")

md("""
## Part 2 · The prankster: build a `Dataset`

A `Dataset` is just a class with **`__len__`** and **`__getitem__`**. PyTorch's `DataLoader` uses those two to make batches.

Our dataset hands out `(rotated_image, k)` where `k ∈ {0,1,2,3}` means "rotated by k × 90°". That `k` is our free label.

`torch.rot90(img, k, dims=(1, 2))` rotates a `(C, H, W)` image by `k` quarter-turns.
""")
md("**TODO #2 — finish `__getitem__`.**")
code(
    """
class RotationPuzzle(Dataset):
    def __init__(self, images, fixed=False):
        self.x = images
        self.fixed = fixed   # fixed=True -> same rotation every time (needed for a fair test set)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        k = i % 4 if self.fixed else random.randint(0, 3)
        rotated = torch.rot90(img, k, dims=(1, 2))
        return rotated, k
""",
    """
class RotationPuzzle(Dataset):
    def __init__(self, images, fixed=False):
        self.x = images
        self.fixed = fixed   # fixed=True -> same rotation every time (needed for a fair test set)

    def __len__(self):
        return ...  # TODO: how many images do we have?

    def __getitem__(self, i):
        img = self.x[i]
        k = i % 4 if self.fixed else random.randint(0, 3)
        rotated = ...  # TODO: rotate img by k quarter turns with torch.rot90 (dims=(1, 2))
        return rotated, k
""",
)
md("✅ **Check** — also a good moment to *look* at what the prankster did.")
code("""
train_ds = RotationPuzzle(train_x)
test_ds  = RotationPuzzle(test_x, fixed=True)

assert len(train_ds) == 50000 and len(test_ds) == 10000
img, k = test_ds[1]
assert img.shape == (3, 32, 32) and k == 1
assert not torch.equal(img, test_x[1]), "image 1 should be rotated!"
print("Dataset ✅")

samples = [test_ds[i] for i in range(16)]
show(torch.stack([s[0] for s in samples]), [f"rot {s[1]*90}°" for s in samples])
""")

md("""
## Part 3 · `DataLoader`

Batches, shuffling, and parallel loading — for free.

**TODO #3 —** make a `train_loader` (batch size 256, **shuffled**) and a `test_loader` (batch size 512, **not** shuffled).
""")
code(
    """
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)
""",
    """
train_loader = ...  # TODO
test_loader  = ...  # TODO
""",
)
md("✅ **Check**")
code("""
xb, yb = next(iter(train_loader))
print("batch of images:", xb.shape, "| batch of labels:", yb.shape, yb[:10])
assert xb.shape == (256, 3, 32, 32) and yb.shape == (256,)
print("DataLoader ✅")
""")

md("""
## Part 4 · The detective: a CNN

You've seen conv layers in the CNN workshop. Here's the blueprint:

```
(3×32×32) → [conv block 32] → [conv block 64] → [conv block 128] → flatten → Linear → ReLU → Linear → 4 scores
```
Each **conv block** = `Conv2d → BatchNorm → ReLU → MaxPool(2)`, and the MaxPool **halves** height & width.

**TODO #4 —** two small things:
1. After 3 conv blocks, how big is the feature map? Fill in the `Linear` input size (channels × H × W).
2. Write `forward`: features → classifier.
""")
code(
    """
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

class RotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),    # 32x32 -> 16x16
            conv_block(32, 64),   # 16x16 -> 8x8
            conv_block(64, 128),  # 8x8   -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),     # 4 classes: 0, 90, 180, 270 degrees
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
""",
    """
def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

class RotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),    # 32x32 -> 16x16
            conv_block(32, 64),   # 16x16 -> 8x8
            conv_block(64, 128),  # 8x8   -> ?x?
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(..., 128),   # TODO: channels * height * width after the features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4),     # 4 classes: 0, 90, 180, 270 degrees
        )

    def forward(self, x):
        # TODO: pass x through self.features, then self.classifier, and return the result
        ...
""",
)
md("✅ **Check** — push a fake batch through; we expect 4 scores per image.")
code("""
model = RotNet().to(device)
out = model(torch.randn(8, 3, 32, 32, device=device))
assert out.shape == (8, 4), f"expected (8, 4) got {tuple(out.shape)}"
n_params = sum(p.numel() for p in model.parameters())
print(f"Model ✅  ({n_params:,} parameters)")
""")

md("""
## Part 5 · Training — the heart of PyTorch

Every PyTorch training step is the same 5-beat rhythm:

| # | what | code |
|---|------|------|
| 1 | forget old gradients | `optimizer.zero_grad()` |
| 2 | forward pass | `logits = model(x)` |
| 3 | measure the error | `loss = loss_fn(logits, y)` |
| 4 | backprop | `loss.backward()` |
| 5 | update weights | `optimizer.step()` |

**TODO #5 —** write those five beats inside the loop.
""")
code(
    """
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen
""",
    """
def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # TODO: the 5 beats
        # 1. zero the gradients
        # 2. logits = forward pass
        # 3. loss = loss_fn(logits, y)
        # 4. backward
        # 5. optimizer step
        logits = ...
        loss = ...

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen
""",
)
code("""
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()                      # turns off dropout, uses BatchNorm running stats
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen
""")
md("""
**TODO #6 —** pick a loss and an optimizer.
- Loss for multi-class classification → `nn.CrossEntropyLoss()`
- Optimizer → `torch.optim.Adam(model.parameters(), lr=1e-3)`
""")
code(
    """
model = RotNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
""",
    """
model = RotNet().to(device)
loss_fn = ...    # TODO
optimizer = ...  # TODO
""",
)
md("""
Before training: with 4 equally likely classes, a model that's just guessing gets **25%** accuracy and a loss of about **ln(4) ≈ 1.39**.
That's the number to beat. 🏁
""")
code("""
l, a = evaluate(model, test_loader, loss_fn)
print(f"before training  ->  loss {l:.3f}  acc {a:.1%}")
""")
code("""
EPOCHS = 8
history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn)
    for k, v in zip(history, (tr_acc, te_acc, tr_loss, te_loss)): history[k].append(v)
    print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.1%} | test loss {te_loss:.3f} acc {te_acc:.1%}")
""")
code("""
plt.figure(figsize=(9, 3))
plt.subplot(1, 2, 1); plt.plot(history["train_loss"], label="train"); plt.plot(history["test_loss"], label="test")
plt.title("loss"); plt.xlabel("epoch"); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history["train_acc"], label="train"); plt.plot(history["test_acc"], label="test")
plt.title("accuracy"); plt.xlabel("epoch"); plt.legend()
plt.tight_layout(); plt.show()
""")
md("""
> 💬 **Discuss with your neighbour:** Is the model over-fitting? What would you look at in the plot to tell?
""")

md("""
## Part 6 · Where does the detective get it wrong?

An overall accuracy hides a lot. Let's break it down **by the original photo class** — is it harder to tell a rotated *frog* or a rotated *ship*?
""")
code("""
@torch.no_grad()
def predict(model, x, bs=1000):
    model.eval()
    return torch.cat([model(x[i:i+bs].to(device)).argmax(1).cpu() for i in range(0, len(x), bs)])

test_rot  = torch.stack([test_ds[i][0] for i in range(len(test_ds))])
test_k    = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
test_cls  = torch.tensor(test_raw.targets)
test_pred = predict(model, test_rot)

for c, name in enumerate(CLASS_NAMES):
    m = test_cls == c
    print(f"{name:>10}: {(test_pred[m] == test_k[m]).float().mean():.1%}")
""")
md("""
**TODO #7 —** show 16 images the model got **wrong**. (Hint: `wrong = (test_pred != test_k).nonzero().flatten()`.)
Can *you* tell which way up they go? 🤔
""")
code(
    """
wrong = (test_pred != test_k).nonzero().flatten()[:16]
show(test_rot[wrong], [f"true {test_k[i]*90}° / said {test_pred[i]*90}°" for i in wrong])
""",
    """
wrong = ...  # TODO: indices where the prediction != the true rotation (first 16 only)
show(test_rot[wrong], [f"true {test_k[i]*90}° / said {test_pred[i]*90}°" for i in wrong])
""",
)

md("""
## Part 7 · 🎉 Fix the album!

Time to pay off the mission. Take 16 photos, **scramble** them, ask the model what rotation it sees, and **undo** it.

If the model says an image is rotated by `k` quarter-turns, we rotate it back by `-k`.
""")
md("**TODO #8 —** un-rotate each image.")
code(
    """
def fix_album(model, imgs):
    \"\"\"imgs: scrambled (N,3,32,32). Returns the un-rotated images.\"\"\"
    pred_k = predict(model, imgs)
    fixed = torch.stack([torch.rot90(img, -int(k), dims=(1, 2)) for img, k in zip(imgs, pred_k)])
    return fixed, pred_k
""",
    """
def fix_album(model, imgs):
    \"\"\"imgs: scrambled (N,3,32,32). Returns the un-rotated images.\"\"\"
    pred_k = predict(model, imgs)
    # TODO: for each image, rotate it back by -k quarter-turns, then stack into one tensor
    fixed = ...
    return fixed, pred_k
""",
)
code("""
idx = torch.randperm(len(test_x))[:16]
originals = test_x[idx]
true_k = torch.randint(0, 4, (16,))
scrambled = torch.stack([torch.rot90(o, int(k), dims=(1, 2)) for o, k in zip(originals, true_k)])

fixed, pred_k = fix_album(model, scrambled)

print("🔀 scrambled");  show(scrambled)
print("🛠️  fixed by your model");  show(fixed)
print("✅ ground truth"); show(originals)
print(f"perfectly restored: {(fixed == originals).flatten(1).all(1).sum().item()}/16")
""")

md("""
---
## 🚀 Stretch challenges (pick any!)

Go as far as you like — these are roughly ordered by difficulty.

**A. Beat your score.** Tweak anything: epochs, learning rate, channels, add a 4th conv block, dropout rate, `Adam` → `AdamW`. Who can get the highest test accuracy? 🏆

**B. Test-time-augmentation (TTA) — a clever trick.** Take a scrambled image, rotate it *extra* by `j = 0,1,2,3` quarter-turns and predict each.
If the true rotation is `k`, the model *should* answer `(k + j) % 4`, so each prediction votes `(pred_j − j) % 4` for `k`.
Average the votes (e.g. average softmax outputs, re-aligned with `torch.roll`) → usually a free accuracy boost. Implement `predict_tta` and compare.

**C. Weak spots.** Which class is hardest? Why? (Think about what a rotated *ball* or *top-down plane* looks like.) What would an even more symmetric dataset do — try **MNIST** (is a rotated `0` or `8` detectable?).

**D. Your own photos.** Upload a photo, resize to 32×32, scramble it and see if the model fixes it. Does it generalise from CIFAR to your holiday pictures?

**E. Level up the puzzle.** Add *mirroring* so there are **8** possible "pranks" (4 rotations × flipped or not). What has to change? (Dataset, last Linear layer, un-doing the prank.)

**F. Peek inside.** Plot the 32 first-layer filters (`model.features[0][0].weight`). What are they looking for?

---
### What you practiced today
`Tensor shapes → Dataset → DataLoader → nn.Module → train loop (zero_grad/forward/loss/backward/step) → eval → using the model`.
That's the skeleton of *every* PyTorch project. Swap the dataset and the model and the rest stays the same. 🎓
""")

# --------------------------------------------------------------------------
STRETCH_B = None


def build(path, idx):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "accelerator": "GPU",
    }
    for cell in CELLS:
        kind, src = cell[0], cell[idx]
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nbf.write(nb, path)
    print("wrote", path)


build("photo_mixup_SOLUTIONS.ipynb", 1)
build("photo_mixup_STUDENT.ipynb", 2)
