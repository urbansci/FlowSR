# Distilling symbolic models from mobility data

This repository contains the implementation and materials of the following paper:
>**Distilling symbolic models from mobility data**   
Hao Guo1†, Weiyu Zhang1†, Junjie Yang, Yuanqiao Hou, Lei Dong∗, Yu Liu∗
>  
>**Abstract:** The search for models that can effectively elucidate complex phenomena remains a fundamental challenge across scientific disciplines, particularly within social sciences, where analytical frameworks derived from first principles are absent. Consequently, to quantify social behavior, researchers often rely on empirical data and statistical models with fixed variables and functional forms, limiting the discovery of more accurate and novel representations. Here, we present a symbolic regression-based study to automatically distill model expressions from large-scale human mobility behavior data. We successfully identify the robust distance decay effect in mobility, as well as various forms of the classical gravity model. Furthermore, our analysis uncovers novel extensions to well-established formulations and demonstrates that the geographic heterogeneity of mobility flows cannot be adequately represented by a universal model. The proposed framework can be broadly applied across the social sciences to uncover the hidden mathematical structure underlying complex social phenomena.

## File Structure
    FlowSR/
    │
    ├── .gitignore              
    ├── LICENSE                 
    ├── README.md              
    │
    ├── Data/
    │   
    ├── Existing_models_evaluation/  
    │
    ├── FlowSR_Julia/               
    │
    └── SymbolicRegression.jl/     

- `Existing_models_evaluation`: This directory contains evaluations of existing models, implemented in Python.
- `SymbolicRegression.jl`: This directory hosts a modified version of the SymbolicRegression.jl package, specifically tailored to search for the allocation mobility models. Please note that this package requires manual installation. Original package can be found [here](   https://github.com/MilesCranmer/SymbolicRegression.jl).
- `FlowSR_Julia`: This directory contains our proposed framework, implemented using Julia and the modified SymbolicRegression.jl package.
- `Data`: This folder contains the publicly available datasets (the US and England datasets) in this study.

## Instructions for running the code
The project is implemented in Julia. Please install Julia from the [official website](https://julialang.org/downloads/).

Julia's robust dependency management system simplifies the process of setting up the project environment. First, clone the repository to your local machine. Then, activate the project environment by running the following commands in terminal:
```julia
cd path/to/your/project/FlowSR_julia
julia # start julia REPL
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Our project is based on `SymbolicRegression.jl`, and we modified it to search for allocation mobility model. The modified package requires manual installation. After activating the project environment in last step, run the following command in julia REPL:
```julia
using Pkg
Pkg.develop(path="path/to/your/project/SymbolicRegression.jl")
```

After preparation of the environment and dependencies, you can run the following command to perform the symbolic regression easily:

```julia
cd path/to/your/project/FlowSR_julia
julia --project="." --threads=4 srflow_us.jl # Take the use for example 
```


## Baseline Evauation
The evaluation of existing models is conducted using Python. Execute the benchmark_allocation.py script directly:
```
python benchmark_allocation.py
```

## Modification Records of SymbolicRegression.jl 

#### OptionsModule

--  **Options** : The construction function is in Options.jl and the definition is in Optionstruct.jl. Attributes `allocation`, `eval_probability`, `ori_sep`, `num_places`,  `optimize_hof` are added.  

-- In allocation mode, `ori_sep` is required as n-dim vector, where n is the number of places; dataset entry `ori_sep[i-1]+1:ori_sep[i]` corresponds to flows with origin `i`. Alternatively, you may input n*n `adjmatrix`, which is transformed into `ori_sep`. `num_places` will be calculated automatically.  

#### LossFunctionsModule

-- **eval_loss**: Generate partition if `allocation`.  
-- **_eval_loss**: Perform probability normalization if `allocation`. If `eval_probability`, do not multiply total outflow.  
-- **batch_sample**: Sample from `1:num_places` instead of `1:dataset.n` if allocation.

#### SymbolicRegressionModule
-- **_equation_search**: if `optimize_hof`, Hall-of-Fame equations will be optimized with entire dataset (even if `batching=true`) after the last `s_r_cycle`.
