# This script is used to generate data and run symbolic regression for Guangdong commuting data at the subdistrict level.
# Due to the different data structures of the county and subdistrict data, we split the SR experiments for the two levels into two scripts.
# `build_x_y_from_raw` specifies whether to build x(input) and y(groundtruth) from raw data or load existing x and y files
# `save_x_y` specifies whether to save x and y files if building x and y from raw data
import MLJ: machine, fit!, predict, report
using SymbolicRegression: SRRegressor
using CSV
using Pickle
using Dates
using NPZ
using DataFrames
using FileIO
using LinearAlgebra, Statistics

select_feat = ["home_pop","work_pop"]

# Specify whether to build x and y from scratch or load existing data
build_x_y_from_raw = false
# Specify whether to save x and y 
save_x_y = false

if build_x_y_from_raw
    # Load the data
    flow_array = npzread(raw"/path/to/gd_commute_flow_matrix_intersubdistrict.npy")
    dist_array = npzread(raw"/path/to/gd_commute_dist_matrix_intersubdistrict.npy")
    id_dict = Pickle.load(raw"/path/to/gd_commute_ids_mapping_intersubdistrict.pkl")
    attr_df = CSV.read(raw"/path/to/gd_commute_attr_intersubdistrict.csv", DataFrame)
    oppo_work_array = npzread(raw"/path/to/gd_commute_opportunity_matrix_intersubdistrict_work_pop.npy")
    oppo_res_array = npzread(raw"/path/to/gd_commute_opportunity_matrix_intersubdistrict_home_pop.npy")

    ori = Int[]
    dest = Int[]
    flow = Float64[]
    dist = Float64[]
    oppo_work = Float64[]
    oppo_res = Float64[]
    featarr = [Float64[] for i in 1:2*size(select_feat, 1)]

    num_regions = size(flow_array, 1)

    ori_count = [0 for i in 1:num_regions]

    for i in 1:num_regions
        for j in 1:num_regions

            # If the flow is zero or skip the diagonal, continue
            if flow_array[i, j] == 0 || i == j
                continue
            end
            
            # Get the origin and destination id
            o_id = id_dict[i-1]
            d_id = id_dict[j-1]

            # Get the attributes of the origin and destination
            if select_feat !== nothing
                o_attr = attr_df[attr_df.street .== o_id, select_feat]
                d_attr = attr_df[attr_df.street .== d_id, select_feat]
            else
                o_attr = attr_df[attr_df.street .== o_id, 3:end]
                d_attr = attr_df[attr_df.street .== d_id, 3:end]
            end
            @assert size(d_attr, 1) == 1
            @assert size(o_attr, 1) == 1

            # Get the distance between the origin and destination
            distance = dist_array[i, j]

            if distance < 0
                println(o_id, " ", d_id, " ", distance)
                println(i, " ", j)
            end

            # Get the opportunity of work and residence
            opportunity_work = oppo_work_array[i, j]
            opportunity_res = oppo_res_array[i, j]

            push!(ori, o_id)
            push!(dest, d_id)
            if logY
                push!(flow, log(flow_array[i, j] + 1))
            else
                push!(flow, flow_array[i, j])
            end
            push!(dist, distance)
            push!(oppo_work, opportunity_work)
            push!(oppo_res, opportunity_res)
            o_attr_vec = Float64.(collect(o_attr[1, :]))
            d_attr_vec = Float64.(collect(d_attr[1, :]))
            for f in 1:size(select_feat, 1)
                push!(featarr[f], o_attr_vec[f])
                push!(featarr[size(select_feat, 1) + f], d_attr_vec[f])
            end
            # println(featarr)
            ori_count[i] += 1
        end
    end

    # Prepare x(input) and y(groundtruth) for SR
    y =  flow

    X = (
        Dist=dist, 
        Oppo_work = oppo_work,
        Oppo_res = oppo_res,
        Ohome=featarr[1], 
        Owork=featarr[2],
        Dhome=featarr[3],
        Dwork=featarr[4],
        )

    ori_sep = [sum(ori_count[1:i]) for i in 1:num_regions]
    if save_x_y
        save("gd_commute_county_X_dist_odwp.jld2", "X", X)
        save("gd_commute_county_Y.jld2", "y", y)
        save("gd_commute_county_sep.jld2", "sep", ori_sep)
    end
else
    # Load the x and y data
    X = load("gd_commute_county_X_dist_odwp.jld2", "X")
    y = load("gd_commute_county_Y.jld2", "y")
    ori_sep = load("gd_commute_county_sep.jld2", "sep")
end

timestamp = Dates.format(Dates.now(), "yyyymmddHHMMSS")[3:end]
model = SRRegressor(
    niterations=100,
    binary_operators=[+, -, *, /, ^],
    unary_operators=[exp, log],
    complexity_of_operators=[exp => 2, log => 2],
    constraints=[(^)=>(-1, 1), exp => 1, log => 1],
    allocation=true,
    optimize_hof=true, 
    ori_sep=ori_sep,
    batching=true,
    batch_size=40,
    output_file="./hall_of_fame_" * timestamp * ".csv",
)

mach = machine(model, X, y) 

fit!(mach)
report(mach)

#yp = predict(mach, X)
# yp = predict(mach, (data=X, idx=2))