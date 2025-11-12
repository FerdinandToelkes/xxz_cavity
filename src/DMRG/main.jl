using ITensors


# Create some example ITensor indices
i = @show Index(3)
j = @show Index(3)
k = @show Index(4)
# l = @show Index(3)

# Create an ITensor with the given indices
A = randomITensor(i, j)
B = randomITensor(i, j)

Ap = prime(A,i)
@show A
@show Ap
@show B

C = Ap * B
@show C

# B = ITensor(j, i)
# C = ITensor(l, j, k)

# D = @show A * B * C