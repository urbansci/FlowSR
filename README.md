# Distilling symbolic models from mobility data

This repository contains the implementation and materials of the following paper:
>**Distilling symbolic models from mobility data**   
Hao Guo†, Weiyu Zhang†, Junjie Yang, Yuanqiao Hou, Lei Dong∗, Yu Liu∗
>  
>**Abstract:** The search for models that can effectively elucidate complex phenomena remains a fundamental challenge across scientific disciplines, particularly in the social sciences, where first-principle analytical frameworks are often lacking. Researchers in these fields frequently rely on empirical data and statistical models with fixed variables and functional forms, which limits the discovery of more accurate and novel representations. Here, we present a symbolic regression-based approach designed to automatically distill model expressions from human mobility data, a crucial aspect of social behavior. Our method successfully identifies the well-established distance decay effect in mobility and various forms of the classical gravity model. Furthermore, we uncover novel extensions to existing models and demonstrates that the geographic heterogeneity of mobility flows necessitates region-specific models rather than universal ones. This framework offers a powerful tool for revealing the underlying mathematical structures of complex social phenomena from observation data.

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
    └── FlowSR_Julia/       

- `Data`: This folder contains the download link of publicly available datasets (the US and England datasets) in this study.
- `Existing_models_evaluation`: This directory contains evaluations of existing models, implemented in Python.
- `FlowSR_Julia`: This directory contains our proposed framework, implemented using Julia and the modified SymbolicRegression.jl package.

## Instructions for running the code
The project is implemented in Julia. Please install Julia from the [official website](https://julialang.org/downloads/).

Julia's robust dependency management system simplifies the process of setting up the project environment. First, clone the repository to your local machine. Then, activate the project environment by running the following commands in terminal:
```julia
cd path/to/FlowSR_Julia
julia # start julia REPL
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Our project is based on `SymbolicRegression.jl`, and we modified it to search for mobility flow allocation model in a friendly forked [repo](https://github.com/urbansci/SymbolicRegression.jl) . 

The modified package requires manual installation. Please clone [modified SymbolicRegression.jl](https://github.com/urbansci/SymbolicRegression.jl) to your local machine. Then run the following command in julia REPL to activate the environment of `FlowSR_julia` and install the modified package:
```julia
cd path/to/FlowSR_Julia
julia
using Pkg
Pkg.activate(".")
Pkg.develop(path="path/to/SymbolicRegression.jl")
```

After preparation of the environment and dependencies, you can run the following command to perform the symbolic regression easily:

```julia
cd path/to/FlowSR_Julia
julia --project="." --threads=4 srflow_us.jl # Take the use for example 
```


## Baseline Evauation
The evaluation of existing models is conducted using Python. Execute the benchmark_allocation.py script directly:
```
python benchmark_allocation.py
```

## Data
The download links of England and US in this study are as follows:
- [England](https://www.dropbox.com/scl/fi/xicio4dlez4fgtx9w9mcw/England.zip?rlkey=s35nev99ztzlc42pbtjcp8e2i&st=tqxbk0wn&dl=0)
- [US](https://www.dropbox.com/scl/fi/61vvp8h9drhw4tihif3ql/US.zip?rlkey=nvu6mvbivl6i7t6jq11h23i5z&st=5daoutgr&dl=0)
- [Beijing-Tianjin-Hebei](https://www.dropbox.com/scl/fi/z68cmg32sphio37isrf83/FlowSR-BTH.zip?rlkey=gyqsil4xky5unp23e6p5rm3kh&st=qebjng47&dl=0)

The downloaded zip files can be extracted to the corresponding folder under `Data/`, and codes can run without any additional modifications.

### England dataset

### US dataset

### Beijing-Tianjin-Hebei (BTH) dataset

## Modification Records of SymbolicRegression.jl 

#### OptionsModule

-  **Options** : Various attributes are added to this class, including `allocation`, `eval_probability`, `ori_sep`, `num_places`,  `optimize_hof`. 

- If `allocation==true`, `ori_sep` is required as n-dim vector, where n is the number of places; dataset entry `ori_sep[i-1]+1:ori_sep[i]` corresponds to flows with origin `i`. Alternatively, you may input n*n `adjmatrix`, which is transformed into `ori_sep`. `num_places` will be calculated automatically.  

#### LossFunctionsModule

- **eval_loss**: Generate partition if `allocation==true`.  
- **_eval_loss**: Perform probability normalization if `allocation==true`. If `eval_probability==true`, do not multiply total outflow.
- **batch_sample**: Sample from `1:num_places` instead of `1:dataset.n` if `allocation==true`.

#### SymbolicRegressionModule
- **_equation_search**: if `optimize_hof==true`, Hall-of-Fame equations will be optimized with entire dataset (even if `batching==true`) after the last `s_r_cycle`.
