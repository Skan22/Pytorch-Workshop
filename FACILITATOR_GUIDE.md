# Facilitator guide: "The Great Photo Mix-Up" (~90 min)

Files
- `photo_mixup_STUDENT.ipynb`: hand out / open in Colab (Runtime -> GPU). 8 TODOs, each followed by a check cell.
- `photo_mixup_SOLUTIONS.ipynb`: yours. Share it after the session.
- `build.py`: regenerates both notebooks from one source (edit here, not the .ipynb files).

| Time | Part | You say / do |
|------|------|--------------|
| 0-5  | Intro | Tell the story. Key idea: we make our own labels (self-supervised). |
| 5-15 | Part 0-1 | Colab setup; TODO 1 (permute/float: the NHWC -> NCHW gotcha everybody hits). |
| 15-30 | Part 2-3 | TODO 2-3. Explain `Dataset` = `__len__` + `__getitem__`; `DataLoader` = batching + shuffle. Why test set is `fixed`. |
| 30-45 | Part 4 | TODO 4. Draw the shapes on the board: 32 -> 16 -> 8 -> 4. Ask the class to compute 128*4*4 before typing. |
| 45-65 | Part 5 | TODO 5-6, then run training (~1-2 min on a T4). Write the 5 beats on the board. Discuss the loss/accuracy plot, baseline 25% / ln4. |
| 65-75 | Part 6 | Per-class accuracy, TODO 7. Pause for "what do you think is hardest?" |
| 75-85 | Part 7 | TODO 8, the payoff: fixing the album. |
| 85-90+ | Stretch | Let people pick; offer a leaderboard for challenge A (best test accuracy). |

Things that commonly go wrong
- Forgetting `.to(device)` or running on CPU: training is slower, still works (lower EPOCHS to 3-4).
- `shuffle=True` on the test loader: harmless here but discuss why we don't.
- Forgetting `optimizer.zero_grad()`: loss still moves but badly. A nice "break it on purpose" demo.
- Forgetting `model.eval()` in evaluation: dropout/BatchNorm behave differently.
- CIFAR download slow: pre-download once and share via Drive, or run it on one machine and screen-share.

Speaker notes for the stretch ideas
- TTA (B): if the image is rotated k, and I rotate it j more, the right answer is (k+j)%4. So every prediction can be shifted back and averaged. It's a nice moment about *using* structure in the problem.
- E (flips -> 8 classes): changes dataset, final Linear(…, 8) and un-doing the prank (flip back, then rotate back; order matters!).

Before you run it live, do a full pass on a Colab GPU and note the final accuracy for your own slide. I tested the notebook logic on a synthetic dataset (no real CIFAR numbers yet).
