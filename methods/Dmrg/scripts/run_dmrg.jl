using Dmrg

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

end

using Random
function main_one()
    Random.seed!(1234)
    L = 4
    N = div(L, 2)
    n_max = 5
    conserve_qns = false
    f_sites = siteinds("Fermion", L; conserve_qns=conserve_qns)
    ph_site = siteinds("Photon", 1; dim=n_max+1, conserve_qns=conserve_qns)
    sites = vcat(f_sites, ph_site)


    # H = xxz_cavity_manual(sites)
    t = 1.0
    U = 0.0
    g = 1.0
    omega = 1.0
    H = xxz_cavity_dev(sites, t, U, g, omega)
    return

    ##############################
    psi0 = MPS(sites)
    # f_states = [isodd(n) ? "1" : "0" for n=1:L]
    f_states = [n<=1 ? "0" : "1" for n=1:L]
    # f_psi0 = random_mps(f_sites, f_states)
    f_psi0 = MPS(f_sites, f_states)

    for i in 1:L
        psi0[i] = f_psi0[i]
    end

    # site L+1: random local state
    psi0[L+1] = randomITensor(sites[L+1])

    # normalize the full MPS
    normalize!(psi0)
    ##############################



    # run DMRG
    nsweeps = 10
    maxdim = [100, 200, 400, 800, 1600]
    cutoff = [1E-10]
    noise = [1E-6, 1E-7, 1E-8, 0.0] # help escape local minima

    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, noise=noise)

    # compute initial energy
    energy_is = inner(psi0', H, psi0)

    @show energy_is
    @show energy



    # apply photon number operator
    n_ph_op = total_photon_number(sites)
    n_ph_is = inner(psi0', n_ph_op, psi0)
    n_ph_gs = inner(psi', n_ph_op, psi)
    @show n_ph_is
    @show n_ph_gs

    # apply fermion number operator
    n_f_op = total_fermion_number(sites)
    n_f_is = inner(psi0', n_f_op, psi0)
    n_f_gs = inner(psi', n_f_op, psi)
    @show n_f_is
    @show n_f_gs


    # compute energy variance -> zero if psi is eigenstate
    var_gs = get_energy_variance(H, psi)
    var_is = get_energy_variance(H, psi0)
    @show var_is
    @show var_gs

end

main_one()

nothing
