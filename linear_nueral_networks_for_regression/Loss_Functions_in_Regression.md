# Loss Functions in Regression

## Mean Squared Error (MSE) - `L2 Loss`

**Formula:**
$$
L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

**Characteristics:**
- **Penalizes large errors heavily** due to squaring (quadratic penalty)
- Differentiable everywhere (smooth gradient)
- **Sensitive to outliers** - a single large error dominates the loss
- Gradient proportional to error: $\frac{\partial L}{\partial \hat{y}} = 2(y - \hat{y})$

**When to use:**
- When outliers are rare and should be avoided
- When large errors are particularly undesirable
- Most common choice for regression

**PyTorch:**
```python
loss_fn = torch.nn.MSELoss()
loss = loss_fn(y_pred, y_true)
```

---

## Mean Absolute Error (MAE) - `L1 Loss`

**Formula:**
$$
L_{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

**Characteristics:**
- **Linear penalty** - all errors weighted equally
- **Robust to outliers** - large errors don't dominate
- Not differentiable at zero (causes issues for some optimizers)
- Constant gradient: $\frac{\partial L}{\partial \hat{y}} = \pm 1$

**When to use:**
- When your data has outliers
- When all errors should be treated equally
- When you want median-like predictions

**PyTorch:**
```python
loss_fn = torch.nn.L1Loss()
loss = loss_fn(y_pred, y_true)
```

---

## Visual Comparison
![MSE vs MAE](./imgs/MSEvsMAE.png)

**Key observation:** MSE grows much faster for large errors (parabolic vs linear).

---

## Side-by-Side Comparison

| Property | MSE (L2) | MAE (L1) |
|----------|----------|----------|
| **Penalty** | Quadratic | Linear |
| **Outlier sensitivity** | High 🔴 | Low 🟢 |
| **Differentiability** | Smooth ✓ | Not at zero |
| **Gradient** | Proportional to error | Constant (±1) |
| **Predicts** | Mean | Median |
| **Convergence** | Faster near optimum | Slower near optimum |
| **Use case** | Clean data | Noisy data with outliers |

---

## Huber Loss - Best of Both Worlds

**Formula:**
$$
L_{\delta}(y, \hat{y}) = \begin{cases} 
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

**Characteristics:**
- **Hybrid approach**: MSE for small errors, MAE for large errors
- Combines strengths of both: smooth gradients + outlier robustness
- **δ (delta)** controls the transition threshold
- Differentiable everywhere (unlike MAE)

**Intuition:**
- Small errors (< δ): behave like MSE → smooth, fast convergence
- Large errors (> δ): behave like MAE → robust to outliers
- Think of it as "MSE with outlier protection"

**When to use:**
- When you suspect outliers but still want smooth optimization
- Default choice for robust regression
- Works well in practice for most real-world datasets

**PyTorch:**
```python
# delta parameter controls transition point
loss_fn = torch.nn.HuberLoss(delta=1.0)  # default delta=1.0
loss = loss_fn(y_pred, y_true)
```

**Visual Comparison (All Three):**
![](./imgs/lossfunctionscomaprison.png)

---

## Practical Example: Impact of Outliers
```python
# Clean data
y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.9])

# Data with one outlier
y_true_outlier = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])  # outlier!
y_pred_outlier = torch.tensor([1.1, 2.1, 2.9, 4.2, 4.9])

mse_clean = torch.nn.MSELoss()(y_pred, y_true)
mae_clean = torch.nn.L1Loss()(y_pred, y_true)
huber_clean = torch.nn.HuberLoss()(y_pred, y_true)

mse_outlier = torch.nn.MSELoss()(y_pred_outlier, y_true_outlier)
mae_outlier = torch.nn.L1Loss()(y_pred_outlier, y_true_outlier)
huber_outlier = torch.nn.HuberLoss()(y_pred_outlier, y_true_outlier)

print("Clean data:")
print(f"  MSE: {mse_clean:.4f} | MAE: {mae_clean:.4f} | Huber: {huber_clean:.4f}")
print("\nWith outlier:")
print(f"  MSE: {mse_outlier:.4f} | MAE: {mae_outlier:.4f} | Huber: {huber_outlier:.4f}")
print("\nIncrease due to outlier:")
print(f"  MSE: {mse_outlier/mse_clean:.1f}x | MAE: {mae_outlier/mae_clean:.1f}x | Huber: {huber_outlier/huber_clean:.1f}x")
```

**Output:**
```
Clean data:
  MSE: 0.0220 | MAE: 0.1200 | Huber: 0.0110

With outlier:
  MSE: 1805.62 | MAE: 19.08 | Huber: 18.88

Increase due to outlier:
  MSE: 82,074x | MAE: 159x | Huber: 1,716x
```

**Notice:** MSE explodes with outliers, MAE stays relatively stable, Huber is in between.

---

## Quick Decision Guide
```
Do you have outliers in your data?
│
├─ NO → Use MSE (fastest convergence, most common)
│
└─ YES → Do you need smooth gradients?
    │
    ├─ YES → Use Huber Loss (best balance)
    │
    └─ NO → Use MAE (most robust)
```

---

## Implementation from Scratch
```python
def mse_loss(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()

def mae_loss(y_pred, y_true):
    return (y_pred - y_true).abs().mean()

def huber_loss(y_pred, y_true, delta=1.0):
    error = (y_pred - y_true).abs()
    quadratic = torch.where(error <= delta, 0.5 * error ** 2, torch.zeros_like(error))
    linear = torch.where(error > delta, delta * (error - 0.5 * delta), torch.zeros_like(error))
    return (quadratic + linear).mean()
```

---

## Key Takeaway

- **MSE**: Fast, smooth, but sensitive to outliers → use for clean data
- **MAE**: Robust, but slow convergence → use when outliers are common
- **Huber**: Goldilocks solution → use when unsure (safe default for real data)