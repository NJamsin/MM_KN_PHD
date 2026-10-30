# m4opt schedule-multi

## For now: only a mono-filter, fixed exposure time version (adoption of the [problem 1](https://m4opt.readthedocs.io/en/latest/design/milp.html#problem-1-fixed-exposure-time) from the doc)
### Limitations:
  - No multivisit > 1 ⇒ No cadence constraint
  - Unique filter implementation ⇒ No filter change constraint

### Constraints:
  - FoV, slew, cuts exactly the same → **mission specific constraints grows linearly with the number of facilities**
  - Coverage constraints are the same, the only difference is that the `pixel_vars` is now shared across all facilities: concatenate all individual mission `good pixels`

### Objective
Same implementation (so $\max \sum_i \rho_i \, p_i$) but with

$$p_i \le \sum_f \sum_{j \in J_i^{(f)}} m_{f,j} \quad \forall i$$

With:
- $\sum_f$ a sum over the different facilities (missions) considered
- $m_{f,j}$ is the $j$-th field of facility $f$
- $J_i^{(f)}$ is the set of indices of fields that contain the pixel $i$ for facility $f$

> ⚠️ **Note:** The only shared var important to the MILP problem is the pixels covered. Hence, pixel probability is only counted once even if multiple telescopes observe it (doesn't prevent the necessary overlap between different telescope tiles).