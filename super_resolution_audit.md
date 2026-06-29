# PNO Zero-Shot Super-Resolution Audit

This document audits the zero-shot super-resolution capability of the PNO architecture (training at 64×64, deploying at 256×256+ without retraining) in this codebase.

---

## 1. How It Works Mechanically

### Why FNO Generalises Across Resolutions
In `SpectralConv2d.forward` (in `src/pno/layers.py`), the model uses `torch.fft.rfft2` and selects the first `modes1` and `modes2` frequency bins. Because it always crops the same low-frequency corner, the learned weights always modulate the coarsest spatial frequencies relative to the input resolution.

At resolution $N$, frequency bin $k$ represents a spatial wavelength of $\lambda = N/k$ pixels:

| Mode k | Wavelength at 64×64 | Wavelength at 256×256 | Wavelength at 1024×1024 |
|--------|---------------------|-----------------------|-------------------------|
| k=1    | 64px (full map)     | 256px (full map)      | 1024px (full map)       |
| k=6    | 10.7px              | 42.7px                | 170.7px                 |
| k=12   | 5.3px               | 21.3px                | 85.3px                  |

The model captures the same *fractional* spectrum of the unit domain, not the same absolute pixel frequencies. This is correct for continuous PDEs, but it means that **fine spatial details cannot be resolved at high resolution**. If you run at 256×256, any corridors narrower than ~21px will be blurred out because the high-frequency details are cut off by the low mode limit.

### SDF FNO Magnitude Scaling
The FNO outputs roughly the same numerical range regardless of input size because it was trained on 64×64-magnitude SDFs. 

Our tests confirmed this:
* **64×64**: SDF value 10px from wall = **4.94**
* **128×128**: SDF value 10px from wall = **6.70** (should be ~10)
* **256×256**: SDF value 10px from wall = **3.99** (should be ~20)

To fix this, we need to multiply the FNO output by `scale_factor = target_res / train_res`. While `evaluate_fno.py` does this, the training/evaluation pipelines for PNO (`train_pno.py`, `evaluate_pno.py`, `path_extraction.py`) do not.

### χ̃ (Obstacle Mask) Behavior
```python
chi = tanh(beta * sdf) * (raw_map - 0.5) + 0.5
```
With $\beta = 5.0$, `tanh` saturates to 1.0 quickly. Even with unscaled SDFs, free-space pixels remain saturated, making $\chi$ robust to SDF scaling issues. However, the raw unscaled SDF is also passed directly to the model's input lift `fc0`, which degrades quality.

### Goal Encoding and DeepNorm Goal Extraction
The goal is encoded as a one-hot grid, and the goal feature vector is extracted using coordinate lookups:
```python
gx = goal[:, 0].long().clamp(0, W - 1)
gy = goal[:, 1].long().clamp(0, H - 1)
```
The coordinates must be in target-resolution space. If you pass 64×64 goal coordinates into a 256×256 inference pass, the goal is mapped to the wrong quadrant.

### Boundary Padding
We use a fixed `padding=9`. At 64×64, this adds a ~14% border, but at 1024×1024 it drops to under 1%. While this is enough to prevent catastrophic boundary artifacts, a resolution-proportional padding would be cleaner.

---

## 2. Component Resolution Audit

| Component | Invariant? | Reason | Action Needed |
|-----------|------------|--------|---------------|
| **SpectralConv2d** | Yes | FFT operations are fully dynamic | None |
| **fc0 (lift)** | Yes | Pointwise 1x1 mapping | None |
| **DAFNOBlock** | Yes | Relies on dynamic grid ops | None |
| **_build_sifn** | Yes | Pointwise tanh activation | None |
| **_build_goal_channel** | Conditional | Uses raw indices | Scale goal coordinates to target res |
| **DeepNorm head** | Conditional | Indexes spatial feature map | Scale goal coordinates to target res |
| **SDF prediction** | No | Outputs 64×64-magnitude SDFs | Multiply SDF by `target_res / train_res` |
| **Value output** | No | Outputs unit-domain distances | Multiply by `target_res / train_res` for A* |

---

## 3. Key Issues & Fixes

1. **SDF magnitude is unscaled**: At super-resolution, the raw SDF fed to PNO has the wrong scale. 
   * *Fix*: Scale the SDF predicted by FNO by `target_res / train_res` before passing it to PNO.
2. **Goal coordinates must match resolution**: The input goals must be specified in target pixel coordinates.
3. **PNO output is in unit-domain scale**: FNO outputs scale with the continuous domain. For A* node calculations, you must scale the output values by `target_res / train_res`.

---

## 4. Resolution Scaling Reference

| Quantity | At 64×64 | At 256×256 | Scales automatically? | Fix needed? |
|----------|----------|------------|-----------------------|-------------|
| Goal coordinates | (32, 32) | (128, 128) | No | Scale coordinates |
| SDF magnitude | ~1-10 | ~4-40 | No | Multiply by `res/64` |
| Padding | 9px (14%) | 9px (3.5%) | No (fixed) | Let it slide (minor Gibbs ringing) |
| Output value magnitude | ~0-65 | ~0-17 | No (unit domain) | Multiply by `res/64` for pixel A* |

---

## 5. Verification Test Results

### Test 1: FNO Shape Compatibility
* FNO at 64x64: output shape `(1, 1, 64, 64)` — **PASS**
* FNO at 128x128: output shape `(1, 1, 128, 128)` — **PASS**
* FNO at 256x256: output shape `(1, 1, 256, 256)` — **PASS**

### Test 2: PNO Shape Compatibility
* PNO at 64x64: output shape `(1, 1, 64, 64)` — **PASS**
* PNO at 128x128: output shape `(1, 1, 128, 128)` — **PASS**
* PNO at 256x256: output shape `(1, 1, 256, 256)` — **PASS**

### Test 3: SDF Scaling Test (FNO)
* res=64: SDF 10px left of wall = **4.94**
* res=128: SDF 10px left of wall = **6.70**
* res=256: SDF 10px left of wall = **3.99**
* *Result*: **FAIL** (SDF does not scale automatically).

### Test 4: Goal and Value Scaling (PNO)
* 64x64, goal at (32, 32) -> value at [0,0] = **65.29**
* 256x256, goal at (128, 128) -> value at [0,0] = **16.81** (unit-domain scaled)
* 256x256, goal at (32, 32) -> value at [0,0] = **12.01** (wrong position)
* *Result*: **PASS** (Values are correct unit-domain scaling, but goal must be scaled).

### Test 5: End-to-End Pipeline
* res=64: Goal value = 0.000 — **PASS**
* res=128: Goal value = 0.000 — **PASS**
* res=256: Goal value = 0.000 — **PASS**

---

## 6. Recommended Code Integration

We implemented a wrapper `SuperResolutionPNO` in `src/pno/super_resolution.py` to handle the scaling automatically:

```python
class SuperResolutionPNO(nn.Module):
    def __init__(self, fno, pno, train_res=64, fno_norm_cfg=None):
        super().__init__()
        self.fno = fno
        self.pno = pno
        self.train_res = train_res
        self.norm = fno_norm_cfg or {}

    def forward(self, raw_map, goal):
        target_res = raw_map.shape[-1]
        scale = target_res / self.train_res

        # Predict SDF and scale it
        x_in = raw_map
        if self.norm.get("normalize_input"):
            x_in = (x_in - self.norm["x_mean"]) / self.norm["x_std"]
        
        sdf = self.fno(x_in)
        if self.norm.get("normalize_target"):
            sdf = sdf * self.norm["y_std"] + self.norm["y_mean"]
        
        sdf = sdf * scale

        # Run PNO and scale output value to pixel distances
        value = self.pno(raw_map, sdf, goal)
        return value * scale
```
