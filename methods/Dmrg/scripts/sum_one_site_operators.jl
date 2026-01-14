using ITensors, ITensorMPS



function mpo_sum_pauli_matrix(L::Int, sites::Vector{<:Index}; pauli::Symbol = :X)::MPO
    pauli ∈ (:X, :Y, :Z) ||
        throw(ArgumentError("pauli must be :X, :Y, or :Z"))

    os::OpSum = OpSum()
    for j::Int in 1:L
        os += 2.0, String(pauli), j
    end

    return MPO(os, sites)
end



let
    L = 3
    sites = siteinds("S=1/2", L)

    # drop the links at the boundaries since they are one dimensional
    links = [Index(2, "link,l=$i") for i in 1:(L-1)]
    @show links
    mpo = mpo_sum_pauli_matrix(L, sites; pauli=:X)
    
    id = [1.0 0.0;
          0.0 1.0]
    X = [0.0 2.0;
         2.0 0.0]
    
    W_bulk_list = []
    for n in 2:(L-1)
        println("Constructing bulk MPO tensor for site $n")
        W_bulk = ITensor(links[n-1], links[n], prime(sites[n]), sites[n]) 
        
        # this can be derived from weighted finite automata
        for i in 1:2, j in 1:2
            W_bulk[1,1,i,j] = id[i,j]
            W_bulk[2,1,i,j] = X[i,j] 
            W_bulk[2,2,i,j] = id[i,j]
        end

        push!(W_bulk_list, W_bulk)
    end
    

    W_1 = ITensor(links[1], prime(sites[1]), sites[1])

    for i in 1:2, j in 1:2
        W_1[1,i,j] = X[i,j]
        W_1[2,i,j] = id[i,j] 
    end

    W_L = ITensor(links[L-1], prime(sites[L]), sites[L])

    for i in 1:2, j in 1:2
        W_L[1,i,j] = id[i,j]
        W_L[2,i,j] = X[i,j] 
    end
     

    # construct MPO manually
    mpo_manual = MPO([W_1; W_bulk_list...; W_L])
    
    random_state = random_mps(sites; linkdims=10)

    for i in 1:L
        println("site $i")
        println(" auto MPO site inds:   ", siteinds(mpo, i))
        println(" manual MPO site inds: ", siteinds(mpo_manual, i))
    end

    for i in 1:L
        #@show random_state[i]
        @show mpo[i]
        @show mpo_manual[i]
    end

    
    diff = mpo * random_state - mpo_manual * random_state
    @show norm(diff)
    @show inner(random_state', mpo, random_state)
    @show inner(random_state', mpo_manual, random_state)
end

# W_L_vec = ITensor(Index(2), Index(1), sites[L], prime(sites[L]))

# for i in 1:2, j in 1:2
#     W_L_vec[1,1,i,j] = X[i,j]
#     W_L_vec[2,1,i,j] = id[i,j] 
# end

# W_1_vec = ITensor(Index(1), Index(2), sites[1], prime(sites[1]))

# for i in 1:2, j in 1:2
#     W_1_vec[1,1,i,j] = id[i,j]
#     W_1_vec[2,1,i,j] = X[i,j] 
# end

