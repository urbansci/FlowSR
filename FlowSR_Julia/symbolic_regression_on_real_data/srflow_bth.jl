# This script is used to generate data and run symbolic regression for Beijing-Tianjin-Hebei data
# `build_x_y_from_raw` specifies whether to build x(input) and y(groundtruth) from raw data or load existing x and y files
# `save_x_y` specifies whether to save x and y files if building x and y from raw data

import MLJ: machine, fit!, predict, report
using SymbolicRegression: SRRegressor
using XLSX
using Dates
using FileIO

level = "county"  # spatial scale
repeat = 5  # number of repeats

# Specify whether to build x and y from scratch or load existing data
build_x_y_from_raw = false

# Specify whether to save x and y 
save_x_y = false

if build_x_y_from_raw
    # First make data
    using Pickle
    if level == "county"
        feat = Dict("pop"=> 9)
        iotype = "io"
    end
    
    select_feat_ori = ["pop"]
    select_feat_dest = ["pop"]
    use_dist = true
    use_io = true  # intervening opportunity
    modified_io = false
    
    flow_dict = Pickle.load("../../Data/BTH/BTH_"* level *"_flow.pkl")
    dist_arr = Float64[]
    dist_dict = Pickle.load("../../Data/BTH/BTH_"* level *"_dist.pkl")
    
    if use_io
        io_arr = Float64[]
        io_dict = Pickle.load("../../Data/BTH/BTH_"* level *"_io.pkl")
    end
    
    units = sort(collect(keys(dist_dict)))
    attrfile = XLSX.readxlsx("../../Data/BTH/BTH_"* level *"_attr.xlsx")
    attrtab = attrfile["attr"]
    
    flow = Float64[]
    geoid2row = Dict{Int,Int}()
    nrows = size(attrtab[:], 1)
    nfeat_ori = size(select_feat_ori,1)
    nfeat_dest = size(select_feat_dest,1)
    nplaces = size(units,1)
    
    ofeatarr = [Float64[] for i in 1:nfeat_ori]
    dfeatarr = [Float64[] for i in 1:nfeat_dest]
    ori_count = [0 for i in 1:nplaces]
    println(nrows-1)
    @assert nplaces==nrows-1
    
    for r in 2:nrows
        geoid2row[attrtab[r, 5]] = r
    end
    
    for o in units
        for d in sort(collect(keys(flow_dict[o])))
            ori_count[geoid2row[o]-1] += 1
            vol = flow_dict[o][d]
            push!(flow, vol)
            oattr = [attrtab[geoid2row[o], feat[f]] for f in select_feat_ori]
            dattr = [attrtab[geoid2row[d], feat[f]] for f in select_feat_dest]
            for f in 1:nfeat_ori
                push!(ofeatarr[f], oattr[f])
            end
            for f in 1:nfeat_dest
                push!(dfeatarr[f], dattr[f])
            end
    
            if use_dist
                dist = dist_dict[o][d]
                push!(dist_arr, dist)
            end
            if use_io
                io = io_dict[o][d]
                if modified_io && iotype=="io"
                    io += attrtab[geoid2row[o], feat["pop"]] 
                end
                push!(io_arr, io)
            end
        end
        if geoid2row[o]%100==0
            println("$(geoid2row[o])/$(nrows-1)")
        end
    end

    # Prepare x(input) and y(groundtruth) for SR
    y = flow
    X = (D=dist_arr,  S=io_arr, Wo=ofeatarr[1], Wd=dfeatarr[1]) 
    ori_sep = [sum(ori_count[1:i]) for i in 1:nplaces]
    
    # Save the data
    if save_x_y
        save("./Data/BTH/BTH_"*level*"_X_dist_io_odp.jld2", "X", X)
        save("./Data/BTH/BTH_"*level*"_Y.jld2", "y", y)
        save("./Data/BTH/BTH_"*level*"_sep.jld2", "sep", ori_sep)
    end
else
    # Load Existing Data
    X = load("./Data/BTH/BTH_"*level*"_X_dist_io_odp.jld2", "X")
    y = load("./Data/BTH/BTH_"*level*"_Y.jld2", "y")
    ori_sep = load("./Data/BTH/BTH_"*level*"_sep.jld2", "sep")
end

for rep in 1:repeat
    timestamp = Dates.format(Dates.now(),"yyyymmddHHMM")[3:end]
    model = SRRegressor(
        niterations=100,
        binary_operators=[+, -, *, /, ^],
        unary_operators=[exp, log],
        complexity_of_operators=[exp => 2, log => 2],
        constraints=[(^)=>(-1, 1), exp => 1, log => 1],
        allocation=true,
        optimize_hof=true, 
        ori_sep=ori_sep,
        output_file="hof_jjj_" *level *"_"* timestamp * ".csv",
    )

    mach = machine(model, X, y)
    fit!(mach)
    report(mach)
end

