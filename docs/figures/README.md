# Report figures

IEEE format: single-column width (3.5 in), Times at 8 pt, no title inside the figure, vector PDF (fonts embedded) for the paper plus 600-dpi PNG. Each series also differs by marker shape or line style, so the figures read in grayscale print.

Regenerate from `docs/results/*.json`:

```bash
python scripts/plot_training_curves.py
```

## Suggested captions

IEEE places the caption **below** the figure, starting with "Fig. N." Renumber to match the paper.

**`fig_val_f1_epochs.pdf`**

> Fig. 1. Validation entity-level F1 after each training epoch. Solid lines are the mean over three random seeds and shaded bands the min–max range across seeds; the dashed line is a single 8-epoch cased run. The uncased model plateaus at 0.984–0.985 from epoch 2, while the cased model is still improving at epoch 5 (0.971) and reaches 0.977 at epoch 7 when trained for 8 epochs. The star marks the checkpoint selected as the final model.

**`fig_loss_curves.pdf`**

> Fig. 2. Training loss (logged every 50 steps) and validation loss (per epoch) for (a) cased and (b) uncased DistilBERT, seed 42, on a log scale. Training loss falls from 1.6 to about 10⁻³. Validation loss flattens after epoch 2, at about 3×10⁻³ for the cased model and 2×10⁻³ for the uncased model, with no rise that would indicate overfitting within five epochs.

## LaTeX

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig_val_f1_epochs.pdf}
  \caption{Validation entity-level F1 after each training epoch. ...}
  \label{fig:val_f1}
\end{figure}
```
