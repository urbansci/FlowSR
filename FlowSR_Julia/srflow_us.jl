# This script is used to generate data and run symbolic regression for US data
# `build_x_y_from_raw` specifies whether to build x(input) and y(groundtruth) from raw data or load existing x and y files
# `save_x_y` specifies whether to save x and y files if building x and y from raw data

import MLJ: machine, fit!, predict, report
using SymbolicRegression: SRRegressor
using XLSX
using Dates
using FileIO

level= "county" # spatial scale ["county"]

# Specify the features to use
select_feat_ori = ["respop", "workpop"] # ["respop", "workpop"]
select_feat_dest = ["respop", "workpop"]
use_dist = true
use_iowork = true  # intervening opportunity, calculated with workpop
use_iores = true  # intervening opportunity, calculated with respop
modified_io = false

# Specify whether to build x and y from scratch or load existing data
build_x_y_from_raw = false

# Specify whether to save x and y 
save_x_y = false

if build_x_y_from_raw

    using Pickle
    if level == "county"
        feat = Dict("respop"=>4, "employedpop"=>5, "workpop"=>6)
        iotype = "io"
    elseif level == "state"
        feat = Dict("respop"=>4, "employedpop"=>5, "workpop"=>6)
        if modified_io
            iotype = "mio"
        else
            iotype = "io"
        end
    elseif level == "csa"
        feat = Dict("respop"=>3, "employedpop"=>4, "workpop"=>5)
        if modified_io
            iotype = "mio"
        else
            iotype = "io"
        end
    end

    flow_dict = Pickle.load("../Data/US/us_acs15_"*level*"_flow.pkl")
    dist_arr = Float64[]
    dist_dict = Pickle.load("../Data/US/us_"*level*"_dist.pkl")

    if use_iowork
        iowork_arr = Float64[]
        iowork_dict = Pickle.load("../Data/US/us_"* level *"_"* iotype *"work.pkl")
    end
    if use_iores
        iores_arr = Float64[]
        iores_dict = Pickle.load("../Data/US/us_"* level *"_"* iotype *"res.pkl")
    end

    units = sort(collect(keys(dist_dict)))
    attrfile = XLSX.readxlsx("../Data/US/us_acs15_"* level *"_attr.xlsx")
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
        geoid2row[parse(Int,attrtab[r, 1])] = r
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
            if use_iores
                iores = iores_dict[o][d]
                if modified_io && iotype == "io"
                    iores += attrtab[geoid2row[o], feat["respop"]] 
                end
                push!(iores_arr, iores)
            end
            if use_iowork
                iowork = iowork_dict[o][d] 
                if modified_io && iotype == "io"
                    iowork += attrtab[geoid2row[o], feat["workpop"]] 
                end
                push!(iowork_arr, iowork)
            end
        end
        if geoid2row[o]%100==0
            println("$(geoid2row[o])/$(nrows-1)")
        end
    end
    y = flow
    #X = (D=dist_arr,  Wo=ofeatarr[1], Wd=dfeatarr[1])
    X = (D=dist_arr, Sr=iores_arr, Sw=iowork_arr, Ro=ofeatarr[1], Wo=ofeatarr[2], Rd=dfeatarr[1], Wd=dfeatarr[2]) # Need to change everytime making new data 

    ori_sep = [sum(ori_count[1:i]) for i in 1:nplaces]
    println(ori_count[1:5])
    println(ori_sep[1:5])

    if save_x_y
        save("./Data/us_"*level*"_X_dist_iorw_odrw.jld2", "X", X)
        save("./Data/us_"*level*"_Y.jld2", "y", y)
        save("./Data/us_"*level*"_sep.jld2", "sep", ori_sep)
    end
else
    X = load("./Data/us_"*level*"_X_dist_iorw_odrw.jld2", "X")
    y = load("./Data/us_"*level*"_Y.jld2", "y")
    ori_sep = load("./Data/us_"*level*"_sep.jld2", "sep")
end

timestamp = Dates.format(Dates.now(),"yyyymmddHHMM")[3:end]
model = SRRegressor(
    niterations=100,
    binary_operators=[+, -, *, /, ^],
    unary_operators=[exp, log],
    complexity_of_operators=[exp => 2, log => 2],
    constraints=[(^)=>(-1, 1), exp => 1, log => 1],
    allocation=true,
    optimizer_algorithm="NelderMead",
    optimize_hof=true, 
    ori_sep=ori_sep,
    output_file="hof_usacs_" *level *"_"* timestamp * ".csv",
    )

if level=="county"
    model.batching=true
    model.batch_size=100
end

mach = machine(model, X, y)
fit!(mach)
report(mach)
    
