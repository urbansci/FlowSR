# This script is used to perform symbolic regression on synthetic **United States** flow data with different types and levels of noise.
# The existing model used to generate the synthetic flow data is specified by parameter `modeltype`.
# Parameter `noisetype` specifies the type of noise, and `sigma` determines the standard deviation of the normal distribution noise introduced to the flow.

import MLJ: machine, fit!, predict, report
push!(LOAD_PATH, "./src")
using SymbolicRegression: SRRegressor
using XLSX
using Dates
using FileIO

# noisetype_list = ["logadd", "mul"]
seed_list = ["1231", "1232", "1233", "1234", "1235"]

for seed in seed_list
    level= "county" # "msoa" or "mlad"
    model = "RM"
    noisetype = "logadd"  # "logadd" "mul"
    stder = "0.75"
    supp = "3"  # remove flow that < supp
    # seed = "6543"

    # First make data
    using Pickle
    if level == "county"
            feat = Dict("respop"=>4, "employedpop"=>5, "workpop"=>6)
    end

    select_feat_ori = ["workpop"]
    select_feat_dest = ["workpop"]
    use_dist = true
    use_iowork = true  # intervening opportunity, calculated with workpop
    use_iores = false  # intervening opportunity, calculated with respop
    modified_io = false

    flow_dict = Pickle.load("../../Data/synthetic/us"* level *"_"* model *"_"* noisetype * stder * "_supp"*supp*"_"*seed*".pkl")
    dist_arr = Float64[]
    dist_dict = Pickle.load("../../Data/US/us_"* level *"_dist.pkl")

    if use_iowork
        iowork_arr = Float64[]
        iowork_dict = Pickle.load("../../Data/US/us_"* level *"_iowork.pkl")
    end
    if use_iores
        iores_arr = Float64[]
        iores_dict = Pickle.load("../../Data/US/us_"* level *"_iores.pkl")
    end

    units = sort(collect(keys(dist_dict)))
    attrfile = XLSX.readxlsx("../../Data/US/us_acs15_"* level *"_attr.xlsx")
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
                if modified_io && level=="county"
                    iores += attrtab[geoid2row[o], feat["respop"]] 
                end
                push!(iores_arr, iores)
            end
            if use_iowork
                iowork = iowork_dict[o][d] 
                if modified_io && level=="county"
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
    X = (D=dist_arr,  Sw=iowork_arr, Wo=ofeatarr[1], Wd=dfeatarr[1])
    # X = (D=dist_arr, Sr=iores_arr, Sw=iowork_arr, Ro=ofeatarr[1], Wo=ofeatarr[2], Rd=dfeatarr[1], Wd=dfeatarr[2]) # Need to change everytime making new data 

    ori_sep = [sum(ori_count[1:i]) for i in 1:nplaces]
    println(ori_count[1:5])
    println(ori_sep[1:5])

    #=
    save("./data/eng_"*level*"_supp3_X_dist_iorw_odrw.jld2", "X", X)
    save("./data/eng_"*level*"_supp3_Y.jld2", "y", y)
    save("./data/eng_"*level*"_supp3_sep.jld2", "sep", ori_sep)


    # Load Existing Data
    X = load("./data/eng_"*level*"_supp3_X_dist_iorw_odrw.jld2", "X")
    y = load("./data/eng_"*level*"_supp3_Y.jld2", "y")
    ori_sep = load("./data/eng_"*level*"_supp3_sep.jld2", "sep")
    =#

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
        output_file="hof_ussynth_" *level *"_"* model *"_"* noisetype * stder * "_supp"*supp*"_"* seed *"_"*timestamp*".csv",
    )


    if level=="county"
        model.batching=true
        model.batch_size=100
    end

    mach = machine(model, X, y)
    fit!(mach)
    report(mach)

    # yp = predict(mach, X)
    # yp = predict(mach, (data=X, idx=2))
end
