using Flux
using CuArrays
using Flux:@treelike
using Distributions
using Base.Iterators:partition

# HYPER PARAMETERS
UP_SAMPLE_FACTOR = 4
UP_SAMPLE_FACTOR_STEP = 2

include("utils.jl")
include("layers.jl")
include("generator.jl")
include("discriminator.jl")

# gen = Gen(1) |> gpu
dis = Discriminator() |> gpu

x = rand(256,256,3,1) |> gpu

out = dis(x)

println(size(out))
