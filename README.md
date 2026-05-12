# fleet-math-py

Core fleet mathematics: **ZHC** (Zero Holonomy Consensus), **H1 emergence detection**, **Laman rigidity**, and **continuous constraint fields**.

Zero external dependencies — pure Python stdlib.

## Modules

### ZHC — Zero Holonomy Consensus (`fleet_math.zhc`)

Models a weighted constraint graph where edge weights represent obstacle-phase differences. A configuration has _zero holonomy_ when the product of weights around every cycle equals 1.

```python
from fleet_math import ConstraintGraph

g = ConstraintGraph()
g.add_edge("A", "B", 1.5)
g.add_edge("B", "C", 0.8)
g.add_edge("C", "A", 0.833)  # 1.5 * 0.8 * 0.833 ≈ 1.0

consensus, violations = g.check_consensus(tolerance=0.01)
# consensus == True
```

### H1 — Emergence Detection (`fleet_math.h1`)

Quantifies graph topological complexity via the first Betti number: β₁ = E − V + C.

```python
from fleet_math import betti_1, emergence_severity, detect_emergence

graph = ...  # any object with .nodes and .edges
b1 = betti_1(graph)
sev = emergence_severity(graph)
emergent = detect_emergence(graph)
```

### Laman — 2D Rigidity (`fleet_math.laman`)

Laman's theorem: a generic 2D bar-joint framework is rigid when each connected component satisfies E ≥ 2V − 3.

```python
from fleet_math import is_rigid, is_minimally_rigid, rigid_margin

is_rigid(graph)           # True if E >= 2V-3 per component
is_minimally_rigid(graph) # True if E == 2V-3 (connected)
rigid_margin(graph)       # E - (2V-3)
```

### Field — Continuous Constraint Field (`fleet_math.field`)

Embeds weighted positions in 2D and interpolates via inverse-distance weighting (IDW).

```python
from fleet_math import Field

f = Field()
f.embed("A", 0.0, 0.0, weight=1.0)
f.embed("B", 1.0, 1.0, weight=0.5)

val = f.read(0.5, 0.5)   # IDW interpolation
grad = f.gradients()      # numerical gradient field
gaps = f.gaps()            # low-density regions
```

## Testing

```bash
python -m pytest tests/
```

## Install

```bash
pip install .
# or
pip install fleet-math
```

## License

MIT

## Polyformalism

FM's `SuperInstance/polyformalism` is the authoritative polyglot constraint kernel — 14 languages (Python, C, Zig, Nim, Rust*...), 3 functions, 2,100 differential test vectors. This library (`fleet-math-py`) extends polyformalism with ZHC consensus, H1 emergence detection, and the continuous field. The polyformalism repo has the base constraint kernel; this repo has the fleet operations built on it.

`constraint_check(lower, upper, values)` → `saturate()` + `Constraint.check()`  
`eisenstein_norm(a, b)` → hexagonal snap in `gatekeeper-as-flux/eisenstein_deadband.py`
