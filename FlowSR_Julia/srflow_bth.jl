import MLJ: machine, fit!, predict, report
push!(LOAD_PATH, "./src")
using SymbolicRegression: SRRegressor
using XLSX
using Dates
using FileIO

level= "county" 
repeat = 5

#= First make data
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

flow_dict = Pickle.load("../data/Jingjinji/JJJ_"* level *"_flow.pkl")
dist_arr = Float64[]
dist_dict = Pickle.load("../data/Jingjinji/JJJ_"* level *"_dist.pkl")

if use_io
    io_arr = Float64[]
    io_dict = Pickle.load("../data/Jingjinji/JJJ_"* level *"_io.pkl")
end

units = sort(collect(keys(dist_dict)))
attrfile = XLSX.readxlsx("../data/Jingjinji/JJJ_"* level *"_attr.xlsx")
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
y = flow
X = (D=dist_arr,  S=io_arr, Wo=ofeatarr[1], Wd=dfeatarr[1])
# Need to change everytime making new data 

ori_sep = [sum(ori_count[1:i]) for i in 1:nplaces]
println(ori_count[1:5])
println(ori_sep[1:5])

save("./data/jjj_"*level*"_X_dist_io_odp.jld2", "X", X)
save("./data/jjj_"*level*"_Y.jld2", "y", y)
save("./data/jjj_"*level*"_sep.jld2", "sep", ori_sep)
exit()

=#

#Load Existing Data
for rep in 1:repeat
    X = load("./data/jjj_"*level*"_X_dist_io_odp.jld2", "X")
    y = load("./data/jjj_"*level*"_Y.jld2", "y")
    ori_sep = load("./data/jjj_"*level*"_sep.jld2", "sep")

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

    # yp = predict(mach, X)
    # yp = predict(mach, (data=X, idx=2))
end

