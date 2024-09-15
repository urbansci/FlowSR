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
- [Beijing-Tianjin-Hebei](https://www.dropbox.com/scl/fi/fx379ra8hsesukn34bl2r/FlowSR-BTH.zip?rlkey=hkoxv7mbft11wa7hxrpth9fq9&st=qx1msb0m&dl=0)

The downloaded zip files can be extracted to the corresponding folder under `Data/`, and codes can run without any additional modifications.

### England dataset

This dataset contains the location of usual residence and place of work of employed residents in England, aggregated to MLAD level, as well as the residential and workplace population of each MLAD. Both the [commuting flow data](https://www.nomisweb.co.uk/census/2011/wu02uk) and the [population data](https://ukdataservice.ac.uk/learning-hub/census/) are collected from the 2011 UK Census.  

- England_mlad_census11_attr.xlsx
    - respop: residential population of each county (in 10^4 persons), collected from [UK Data Service](https://ukdataservice.ac.uk/learning-hub/census/)
    - workpop: workplace population of each county (in 10^4 persons), collected from [UK Data Service](https://ukdataservice.ac.uk/learning-hub/census/)
    - centx/centy: projected coordinates of polygon centroid for each county (in meters, EPSG: 27700), calculated from the [official boundary shapefile in 2011](https://geoportal.statistics.gov.uk/datasets/ons::census-merged-local-authority-districts-december-2011-boundaries-gb-bgc/about)
- England_mlad_census11_supp3.pkl: number of commuters from each origin (residence) MLAD to each destination (work) MLAD
- England_mlad_dist.pkl: spherical distances between each pair of MLADs, calculated based on the projected coordinates
- England_mlad_iores.pkl: intervening opportunites from each origin MLAD to each destination MLAD, calculated based on the residential population
- England_mlad_iowork.pkl: intervening opportunites from each origin MLAD to each destination MLAD, calculated based on the workplace population

### US dataset

This dataset contains the location of usual residence and place of work of employed residents in the Contiguous US, aggregated to the county level, as well as the residential and workplace population of each county. Both the [commuting flow data](https://www.census.gov/data/tables/2015/demo/metro-micro/commuting-flows-2015.html) and the [population data](https://data.census.gov/) are collected from the American Community Survey (2011-2015 ACS 5-year estimate). 

- us_acs15_county_attr.xlsx
    - respop: residential population of each county (in 10^4 persons), collected from [Census.gov](https://data.census.gov/)
    - workpop: workplace population of each county (in 10^4 persons), calculated from the commuting flow data (workers from Contiguous US only)
    - centx/centy: longitude/latitude of polygon centroid for each county (in degrees), calculated from the [official boundary shapefile in 2015](https://www.census.gov/geographies/mapping-files/2015/geo/carto-boundary-file.html)
- us_acs15_county_flow.pkl: number of commuters from each origin (residence) county to each destination (work) county
- us_county_dist.pkl: spherical distances between each pair of counties, calculated based on the geographical coordinates
- us_county_iores.pkl: intervening opportunites from each origin county to each destination county, calculated based on the residential population
- us_county_iowork.pkl: intervening opportunites from each origin county to each destination county, calculated based on the workplace population

### Beijing-Tianjin-Hebei (BTH) dataset

This dataset contains aggregated inter-county human mobility flows from November 4 to November 10, 2019, provided by China Unicom, as well as population for each county in 2019, collected from official statistical yearbooks. 

- BTH_county_attr.xlsx
    - pop: household registered population in 2019 (in 10^4 persons), collected from 2020 offical statistical yearbooks
    - lon/lat: longitude/latitude of each county, retrieved from the Geocoder API provided by [amap.com](https://amap.com/)
- BTH_county_flow.pkl: number of movements from each origin county to each destination county, provided by China Unicom
- BTH_county_dist.pkl: spherical distances between each pair of counties, calculated based on the geographical coordinates
- BTH_county_io.pkl: intervening opportunites from each origin county to each destination county, calculated based on the household population

You can read the `.pkl` files above using the `pickle` package in Python (or the `Pickle` package in Julia). An example in Python:
```python
flow_file = open("../Data/US/us_acs15_county_flow.pkl", 'rb')
flow_dict = pickle.load(flow_file)
```
The file structure is a nested dictionary. `flow_dict[ID_of_A][ID_of_B]` is the flow volume from county A to county B. The usage for distance or intervening opportunities is similar. The spatial unit IDs are defined as  

- England: `MLAD_CODE` in England_mlad_census11_attr.xlsx, removing the prefix `E41`. 
- US: `GEOID` in us_acs1_county_attr.xlsx
- Beijing-Tianjin-Hebei: 6-digit `code` in BTH_county_attr.xlsx

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
