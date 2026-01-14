using ITensors, ITensorMPS

ITensors.enable_debug_checks()


function xxz_chain_mpo(L::Int, sites::Vector; t::Float64, U::Float64)::MPO
    os = OpSum()
    for j=1:L-1
        os += -t,"c†",j,"c",j+1
        os += +t,"c",j,"c†",j+1
        os += U,"n",j,"n",j+1
    end
    H = MPO(os, sites)
    return H
end

function main()
    L = 3
    t = 3.0
    U = 2.0
    # sites = siteinds("Fermion", L; conserve_qns=false)
    sites = siteinds("Fermion", L)
    
    # build the Hamiltonian MPO using OpSum
    H = xxz_chain_mpo(L, sites; t=t, U=U)
    random_state = random_mps(sites; linkdims=10)
    

    # build it by hand to compare
    w = [Index(5, "MPO") for _ in 1:(L+1)]

    @show H * random_state
    @show inner(random_state', H, random_state)

    # @show H
    # for i in 1:L
    #   W = H[i]
    #   @show W
    # end
    return
    # initialize a MPS for a system at half-filling
    state = [isodd(n) ? "Occ" : "Emp" for n=1:L]
    psi0 = productMPS(sites, state)
    @show flux(psi0)
    # psi0 = random_mps(sites; linkdims=10)

    nsweeps = 5
    maxdim = [10,20,100,100,200]
    cutoff = [1E-10]

    energy,psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

    @show psi


end

main()
