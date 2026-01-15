using ITensors, ITensorMPS

ITensors.enable_debug_checks()

function heisenberg_mpo(N::Int, sites::Vector)::MPO
  os = OpSum()
  for j=1:N-1
    os += "Sz",j,"Sz",j+1
    os += 1/2,"S+",j,"S-",j+1
    os += 1/2,"S-",j,"S+",j+1
  end
  H = MPO(os, sites)
  return H
end

function sum_of_two_one_site_operators(N::Int, sites::Vector)::MPO
  os = OpSum()
  for j=1:N
    os += 1.0, "Y", j
    os += 1.0, "X", j
  end
  H = MPO(os, sites)
  return H
end

function main()
  L = 3
  sites = siteinds("S=1/2", L)
  
  # H = heisenberg_mpo(L, sites)
  H = sum_of_two_one_site_operators(L, sites)
  
  for i in 1:L
    W = H[i]
    @show W
  end

  
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
