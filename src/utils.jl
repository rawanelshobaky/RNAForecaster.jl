#utility functions for RNAForecaster

#filtering expression count matrices
"""
`filterByZeroProp(t1Counts::Matrix{Float32}, t2Counts::Matrix{Float32},
    zeroProp::Float32)`

Filter by zero proportion for both genes and cells. Very high sparsity prevents
the neural network from achieving a stable solution.

# Required Arguments
* t1Counts - Counts matrix for time 1. Should be genes x cells and Float32.
* t2Counts - Counts matrix for time 2. Should be genes x cells and Float32.

# Keyword Arguments
* zeroProp - proportion of zeroes allowed for a gene or a cell. 0.98f0 by default
"""
function filterByZeroProp(t1Counts::Matrix{Float32}, t2Counts::Matrix{Float32};
    zeroProp::Float32 = 0.98f0)

    zeroPropT1Genes = Array{Float32}(undef, size(t1Counts)[1])
    for i=1:size(t1Counts)[1]
        zeroPropT1Genes[i] = length(findall(x->x == 0, t1Counts[i,:]))/size(t1Counts)[2]
    end
    zeroPropT2Genes = Array{Float32}(undef, size(t2Counts)[1])
    for i=1:size(t2Counts)[1]
        zeroPropT2Genes[i] = length(findall(x->x == 0, t2Counts[i,:]))/size(t2Counts)[2]
    end

    t1Sub = t1Counts[intersect(findall(x->x < zeroProp, zeroPropT1Genes),
     findall(x->x < zeroProp, zeroPropT2Genes)),:]
    t2Sub = t2Counts[intersect(findall(x->x < zeroProp, zeroPropT1Genes),
     findall(x->x < zeroProp, zeroPropT2Genes)),:]

     zeroPropT1Cells = Array{Float32}(undef, size(t1Counts)[2])
     for i=1:size(t1Counts)[2]
         zeroPropT1Cells[i] = length(findall(x->x == 0, t1Counts[:,i]))/size(t1Counts)[1]
     end
     zeroPropT2Cells = Array{Float32}(undef, size(t2Counts)[2])
     for i=1:size(t2Counts)[2]
         zeroPropT2Cells[i] = length(findall(x->x == 0, t2Counts[:,i]))/size(t2Counts)[1]
     end

     t1Sub = t1Sub[:,intersect(findall(x->x < zeroProp, zeroPropT1Cells),
      findall(x->x < zeroProp, zeroPropT2Cells))]
     t2Sub = t2Sub[:,intersect(findall(x->x < zeroProp, zeroPropT1Cells),
      findall(x->x < zeroProp, zeroPropT2Cells))]

     return (t1Sub, t2Sub)
end


"""
`filterByGeneVar(t1Counts::Matrix{Float32}, t2Counts::Matrix{Float32},
    topGenes::Int)`

Filter by gene variance, measured across both count matrices.

# Required Arguments
* t1Counts - Counts matrix for time 1. Should be genes x cells and Float32.
* t2Counts - Counts matrix for time 2. Should be genes x cells and Float32.
* topGenes - the number of top most variable genes to use

# Examples
`mat1 = Float32.(randn(50,50))
mat2 = Float32.(randn(50,50))
hvgFilteredCounts = filterByGeneVar(mat1, mat2, 20)`
"""
function filterByGeneVar(t1Counts::Matrix{Float32}, t2Counts::Matrix{Float32},
    topGenes::Int)

    geneVars = vec(var(t1Counts + t2Counts, dims = 2))
    ordered = sortperm(geneVars)

    t1Counts = t1Counts[ordered[1:topGenes],:]
    t2Counts = t2Counts[ordered[1:topGenes],:]

    return (t1Counts, t2Counts)
end

    