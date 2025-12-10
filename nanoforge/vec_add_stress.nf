n = 10000
sz = 80000
A = alloc(sz)
B = alloc(sz)
C = alloc(sz)

i = 0
init_start:
if i == n goto init_done
A[i] = i
B[i] = i
i = i + 1
goto init_start

init_done:

iter = 0
iter_max = 100

stress_loop:
if iter == iter_max goto end

i = 0
inner_loop:
if i == n goto inner_end

v1 = A[i]
v2 = B[i]
sum = v1 + v2
C[i] = sum
i = i + 1

goto inner_loop

inner_end:
iter = iter + 1
goto stress_loop

end:
val = C[10]
free(A)
free(B)
free(C)
return val
