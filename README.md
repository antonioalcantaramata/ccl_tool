# ccl_tool

Chance Constraint Learning (CCL): a complete framework for chance constraint learning is developed to allow practitioners to solve chance constraints and also adding probabilistic guarantees over learned constraints.

Note: Most of the code for point estimation is derived from the work in [https://github.com/donato-maragno/OptiCL](https://github.com/donato-maragno/OptiCL)

## Usage

Initialization:

```
from CCL import CCL
ccl_tool = CCL(X, y, methodology, p_model, q=None, M_super=None, side=None)
