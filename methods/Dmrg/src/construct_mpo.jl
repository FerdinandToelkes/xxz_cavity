using ITensors, ITensorMPS







# the following functions are not used, but were merely used to 
# test how ITensors works

function heisenberg_mpo(sites::Vector{<:Index})::MPO
    L = length(sites)
    os = OpSum()
    for j=1:L-1
        os += "Sz",j,"Sz",j+1
        os += 1/2,"S+",j,"S-",j+1
        os += 1/2,"S-",j,"S+",j+1
    end
    H = MPO(os, sites)
    return H
end

function heisenberg_manual_mpo(sites::Vector{<:Index})::MPO

end


"""
    pauli_matrix_sum_mpo(sites::Vector{<:Index}, a::Float64; pauli::Symbol=:X)::MPO

Construct an MPO representing the sum of Pauli matrices of type `pauli` over `sites`.

The corresponding operator is given by:
    ``H = a * Σ_{j=1}^{L} σ_j``

Throw an `ArgumentError` if `pauli` is not one of `:X`, `:Y`, or `:Z`.
Throw an `ArgumentError` if the `sites` are not for spin-1/2 particles.
"""
function pauli_matrix_sum_mpo(sites::Vector{<:Index}, a::Float64; pauli::Symbol=:X)::MPO
    for s in sites
        hastags(s, "S=1/2") ||
            throw(ArgumentError("All site indices must be for spin-1/2 particles"))
    end
    
    pauli ∈ (:X, :Y, :Z) ||
        throw(ArgumentError("pauli must be :X, :Y, or :Z"))

    os = OpSum()
    for j in 1:length(sites)
        os += a, String(pauli), j
    end

    return MPO(os, sites)
end


"""
    pauli_matrix_sum_manual_mpo(sites::Vector{<:Index}, a::Float64; pauli::Symbol=:X)::MPO

Construct an MPO representing the sum of Pauli matrices of type `pauli` over `sites` by hand.

The corresponding operator is given by:
    ``H = a * Σ_{j=1}^{L} σ_j``

Throw an `ArgumentError` if `pauli` is not one of `:X`, `:Y`, or `:Z`.
Throw an `ArgumentError` if the `sites` are not for spin-1/2 particles.
"""
function pauli_matrix_sum_manual_mpo(sites::Vector{<:Index}, a::Float64; pauli::Symbol=:X)::MPO
    for s in sites
        hastags(s, "S=1/2") ||
            throw(ArgumentError("All site indices must be for spin-1/2 particles"))
    end
    
    if pauli == :X
        pauli_matrix = [0.0 1.0; 1.0 0.0]
    elseif pauli == :Y
        pauli_matrix = [0.0 -im; im  0.0]
    elseif pauli == :Z
        pauli_matrix = [1.0  0.0; 0.0 -1.0]
    else
        throw(ArgumentError("pauli must be :X, :Y, or :Z"))
    end

    id = [1.0 0.0; 0.0 1.0]

    L = length(sites)

    # drop the links at the boundaries since they are one dimensional
    links = [Index(2, "link,l=$i") for i in 1:(L-1)]

    W_bulk_list = []
    for n in 2:(L-1)
        W_bulk = ITensor(links[n-1], links[n], prime(sites[n]), sites[n]) 
        
        # this can be derived from weighted finite automata
        for i in 1:2, j in 1:2
            W_bulk[1,1,i,j] = id[i,j]
            W_bulk[2,1,i,j] = a*pauli_matrix[i,j] 
            W_bulk[2,2,i,j] = id[i,j]
        end

        push!(W_bulk_list, W_bulk)
    end
    

    W_1 = ITensor(links[1], prime(sites[1]), sites[1])

    for i in 1:2, j in 1:2
        W_1[1,i,j] = a*pauli_matrix[i,j]
        W_1[2,i,j] = id[i,j] 
    end

    W_L = ITensor(links[L-1], prime(sites[L]), sites[L])

    for i in 1:2, j in 1:2
        W_L[1,i,j] = id[i,j]
        W_L[2,i,j] = a*pauli_matrix[i,j] 
    end

    return MPO([W_1; W_bulk_list...; W_L])
end
