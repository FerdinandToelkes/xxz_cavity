using ITensors

# Define indices
i = Index(3)
j = Index(3)
k = Index(3)

T = randomITensor(i, j, k)
Q,R = qr(T, (i, j))

@show T
@show Q
@show R

@assert isapprox(Q*R, T)


