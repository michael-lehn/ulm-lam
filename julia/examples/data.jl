# The bike sharing data for the Julia showcases (ridge.jl, bayes.jl).

using DelimitedFiles

"""The bike sharing data as in gp.py: 14 standardised features, log(1 +
rentals) standardised (UCI; downloaded by python/examples/gp.py)."""
function bike_data()
    zip = joinpath(@__DIR__, "..", "..", "build", "data", "bike_sharing.zip")
    isfile(zip) || error("run python3 python/examples/gp.py first (downloads the data)")
    d, head = readdlm(IOBuffer(read(`unzip -p $zip hour.csv`, String)), ',', Any; header=true)
    col(name) = Float64.(d[:, findfirst(==(name), vec(head))])
    cyc(v, p) = (sin.(2pi .* v ./ p), cos.(2pi .* v ./ p))
    X = hcat(cyc(col("hr"), 24)..., cyc(col("mnth"), 12)..., cyc(col("weekday"), 7)...,
             col("yr"), col("workingday"), col("holiday"), col("weathersit"), col("temp"),
             col("atemp"), col("hum"), col("windspeed"))
    X = (X .- sum(X; dims=1) ./ size(X, 1)) ./ std_(X)
    y = log1p.(col("cnt"))
    X, (y .- sum(y) / length(y)) ./ std_(y)
end
std_(x::AbstractVector) = sqrt(sum(abs2, x .- sum(x) / length(x)) / length(x))
std_(X::AbstractMatrix) = mapslices(std_, X; dims=1)

