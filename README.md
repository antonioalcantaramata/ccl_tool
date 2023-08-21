# ccl_tool

Chance Constraint Learning (CCL): a complete framework for chance constraint learning is developed to allow practitioners to solve chance constraints and also adding probabilistic guarantees over learned constraints.

Note: Most of the code for point estimation is derived from the work in [https://github.com/donato-maragno/OptiCL](https://github.com/donato-maragno/OptiCL)

## Usage

### Initialization

After cloning or downloading **all** the .py files in your working environment, you can initialize the ccl_tool.

```
from CCL import CCL
ccl_tool = CCL(X, y, methodology, p_model, q=None, M_super=None, side=None)
```

- X and y represents the independent and dependent variable in our dataset, respectively. 
- methodology is one of ['point', 'quantile', 'superquantile'] 
- p_model is one of ['lr', 'svm', 'tree', 'rf', 'gbm', 'nn']
- q is used to fit the quantile when using 'quantile' or 'superquantile' as methodology, and ranges in (0,1)
- M_super is the number of quantiles used to approximate the superquantile (only when p_model is not a tree-based method)
- side in ['left', 'right'] indicates the side of the quantile to use when computing the superquantile

### Training

```
model = ccl_tool.train()
```

We save the trained p_model. In most cases, a hyper-parameter selection is applied.

### Constraint learning

```
cons = ccl_tool.constraint_build(model)
```

Linear (sometimes with binary variables) constraints are generated to represent the p_model and be embedded within our MIO problem.

### Constraint embedding

```
conceptual_model= init_conceptual_model(cost_p)
ccl_tool.const_embed(conceptual_model, cons, 'y_name', lb, ub)
```

conceptual_model represents our initialized MIO problem in Pyomo. Generated constraints (cons) are embedded in the model (conceptual_model). A name for the learned variables is needed ('y_name'), and an additional lower and/or upper bound for the chance constraint (lb, ub).

## Citation

If you use this code, you can cite it in you work as:

```
@article{alcantara2023data,
  title={On data-driven chance constraint learning for mixed-integer optimization problems},
  author={Alc{\'a}ntara, Antonio and Ruiz, Carlos},
  journal={Applied Mathematical Modelling},
  volume={121},
  pages={445--462},
  year={2023},
  publisher={Elsevier}
}
```
