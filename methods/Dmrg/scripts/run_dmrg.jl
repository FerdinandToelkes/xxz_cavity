# using Dmrg
using ITensors
using ITensorMPS
using LinearAlgebra





function main()
    L = 3  # Number of sites
    N_f = div(L, 2)  # half-filling
    N_ph = 2  # Max number of photons
    t = 1.0  # Hopping parameter
    U = 2.0 * t  # On-site interaction
    g = 0.5 * t / sqrt(L) # Light-matter coupling
    omega = 1.0  # Cavity frequency
    f_sites = siteinds("Fermion", L; conserve_qns=true)
    # b_site = siteind("Boson"; dim=N_ph+1, conserve_qns=true)
    # b_site = Index(QN() => (N_ph+1); tags="Boson,Site,n=$(L+1)")
    # Create a boson site index with exactly one QN block
    b_site = Index(QN(0) => N_ph+1; tags="Boson")
    # i = Index(QN(0)=>2,QN(1)=>3;tags="i")


    sites = vcat(f_sites, b_site)

    # L fermionic sites (conserved particle name) + 1 bosonic site (no conservation)
    H = xxz_cavity(sites, t, U, g, omega)

    return
    f_state = [isodd(n) ? "1" : "0" for n=1:L]
    b_state = "0"
    state = vcat(f_state, [b_state])
    psi0 = MPS(sites,state)

    nsweeps = 10
    maxdim = [10,20,100,100,200] #1000
    cutoff = [1E-10]


    @time energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

    # check if psi is actually an eigenstate
    H2 = inner(H,psi,H,psi)
    E = inner(psi',H,psi)
    var = H2-E^2
    @show var

    # compute number of fermions
    N_f_op = total_fermion_number(sites)
    n_f = inner(psi', N_f_op, psi)
    n_ph = total_photon_number(sites)
    n_ph_val = inner(psi', n_ph, psi)
    println("Ground state energy = $(energy)")
    println("Number of fermions = $(n_f)")
    println("Number of photons = $(n_ph_val)")

    # # Define model parameters
    # model = HeisenbergModel(10; J=1.0)

    # # Set DMRG parameters
    # dmrg_params = DmrgParameters(max_bond_dim=100, num_sweeps=10)

    # # Run DMRG simulation
    # result = run_dmrg(model, dmrg_params)

    # # Output results
    # println("Ground state energy: ", result.ground_state_energy)
    # println("Entanglement entropy: ", result.entanglement_entropy)
end


main()
nothing
