n = 10000
sz = 80000
A = alloc(sz)
B = alloc(sz)
C = alloc(sz)

i = 0
one = 1

loop:
if i == n goto end

A[i] = i
B[i] = i

v1 = A[i]
v2 = B[i]

sum = v1 + v2

C[i] = sum

i = i + one
goto loop

end:
idx = 10
res = C[idx]
free(A)
free(B)
free(C)
return res
