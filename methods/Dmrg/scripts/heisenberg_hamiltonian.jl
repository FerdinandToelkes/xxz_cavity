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

  # construct MPO by hand to compare
  w = [Index(2, "MPO") for _ in 1:(L+1)]
  W = ITensor(w[1], w[2]', sites[1]', sites[1])
  W[w[1]=>1, w[2]=>1] += op("Id", sites[1])
  # W[w[1]=>1, w[2]=>2] = op("X", sites[1])
  # W[w[1]=>2, w[2]=>1] = op("Y", sites[1])
  # W[w[L]=>1, w[L+1]=>1] = op("Id", sites[L])
  # W[w[L]=>2, w[L+1]=>1] = op("X", sites[L]) + op("Y", sites[L])
  # for j=2:L-1
  #   W = ITensor(w[j], w[j+1]', sites[j]', sites[j])
  #   W[w[j]=>1, w[j+1]=>1] = op("Id", sites[j])
  #   W[w[j]=>1, w[j+1]=>2] = op("X", sites[j])
  #   W[w[j]=>2, w[j+1]=>1] = op("Y", sites[j])
  # end
  H_manual = MPO(W for W in w[1:end-1])
  

  # compare by applying to a random state
  random_state = random_mps(sites; linkdims=10)
  @show H * random_state
  @show inner(random_state', H, random_state)
  @show H_manual * random_state
  @show inner(random_state', H_manual, random_state)
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
