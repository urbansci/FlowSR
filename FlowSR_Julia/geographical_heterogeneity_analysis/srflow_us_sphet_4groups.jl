# This script is used to generate data and run symbolic regression for heterogeneity analysis on US data (4 groups based on residential region)

import MLJ: machine, fit!, predict, report
using SymbolicRegression: SRRegressor
using XLSX
using Dates
using FileIO
using Pickle

level= "county" # spatial scale

if level == "county"
    feat = Dict("respop"=>4, "employedpop"=>5, "workpop"=>6)
end

# look up table for state-region relationships
region_state = Dict(42=> 1, 50=> 1, 25=> 1, 33=> 1, 9=> 1, 34=> 1, 23=> 1, 44=> 1, 36=> 1, 31=> 2, 46=> 2, 18=> 2, 17=> 2, 19=> 2, 39=> 2, 55=> 2, 29=> 2, 26=> 2, 20=> 2, 27=> 2, 38=> 2, 53=> 4, 35=> 4, 8=> 4, 49=> 4, 56=> 4, 32=> 4, 30=> 4, 4=> 4, 41=> 4, 16=> 4, 6=> 4, 21=> 3, 13=> 3, 5=> 3, 28=> 3, 47=> 3, 45=> 3, 1=> 3, 37=> 3, 40=> 3, 51=> 3, 54=> 3, 22=> 3, 12=> 3, 10=> 3, 48=> 3, 24=> 3, 11 => 3)

# Specify the features to use
select_feat_ori = ["workpop"] # subset of ["respop", "workpop"]
select_feat_dest = ["workpop"] # subset of ["respop", "workpop"]
use_dist = true
use_iowork = true  # intervening opportunity, calculated with workpop
use_iores = true  # intervening opportunity, calculated with respop

flow_dict = Pickle.load("../../Data/US/us_acs15_"*level*"_flow.pkl")
dist_dict = Pickle.load("../../Data/US/us_"*level*"_dist.pkl")
if use_iowork
    iowork_dict = Pickle.load("../../Data/US/us_"* level *"_iowork.pkl")
end
if use_iores
    iores_dict = Pickle.load("../../Data/US/us_"* level *"_iores.pkl")
end

units = sort(collect(keys(dist_dict)))
attrfile = XLSX.readxlsx("../../Data/US/us_acs15_"* level *"_attr.xlsx")
attrtab = attrfile["attr"]

nrows = size(attrtab[:], 1)
nfeat_ori = size(select_feat_ori,1)
nfeat_dest = size(select_feat_dest,1)
nplaces = size(units, 1)

geoid2row = Dict{Int,Int}()
for r in 2:nrows
    geoid2row[parse(Int,attrtab[r, 1])] = r
end

for region in 1:4
    flow = Float64[]
    dist_arr = Float64[]
    iowork_arr = Float64[]
    iores_arr = Float64[]
    ofeatarr = [Float64[] for i in 1:nfeat_ori]
    dfeatarr = [Float64[] for i in 1:nfeat_dest]
    ori_sep = Int[]
    flow_count = 0
    println(nrows-1)  # number of spatial units
    @assert nplaces==nrows-1

    for o in units
        if geoid2row[o]%100==0
            println("$(geoid2row[o])/$(nrows-1)")  # progress of importing data
        end
        if region_state[div(o, 1000)]!=region
            continue
        end
        for d in sort(collect(keys(flow_dict[o])))
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
                push!(iores_arr, iores)
            end
            if use_iowork
                iowork = iowork_dict[o][d] 
                push!(iowork_arr, iowork)
            end
        end
        flow_count += size(collect(keys(flow_dict[o])), 1)
        push!(ori_sep, flow_count)
    end
    y = flow
    X = (D=dist_arr, Sr=iores_arr, Sw=iowork_arr, Ro=ofeatarr[1], Wo=ofeatarr[2], Rd=dfeatarr[1], Wd=dfeatarr[2]) # need to remove features that are not used 
    
    # Run symbolic regression
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
        output_file="hof_usacs_" *level *"_reg"*string(region)*"_"* timestamp * ".csv",
    )

    if level=="county"
        model.batching=true
        model.batch_size=100
    end

    mach = machine(model, X, y)
    fit!(mach)
    report(mach)

end

