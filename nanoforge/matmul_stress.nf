fn main() {
    N = 75
    # Matrix A: N x N
    # Matrix B: N x N
    # Result C: N x N
    # Memory Layout (Elements, not bytes):
    # A: [0, N*N)
    # B: [N*N, 2*N*N)
    # C: [2*N*N, 3*N*N)
    
    elems_per_matrix = N * N
    
    # Alloc takes BYTES
    bytes_per_matrix = elems_per_matrix * 8
    total_size = bytes_per_matrix * 3
    mem = alloc(total_size)
    
    # Initialize Matrices
    # A[i,j] = i + j
    # B[i,j] = i - j
    
    i = 0
    label init_loop_i
    if i == N goto init_loop_i_end
    
        j = 0
        label init_loop_j
        if j == N goto init_loop_j_end
        
            # Offset A (Element Index) = i * N + j
            offset = i * N
            offset = offset + j
            
            # A[offset] = i + j
            val_a = i + j
            
            # Offset B (Element Index) = offset + elems_per_matrix
            offset_b = offset + elems_per_matrix
            
            # B[offset] = i - j
            val_b = i - j
            
            mem[offset] = val_a
            mem[offset_b] = val_b
            
            j = j + 1
            goto init_loop_j
        label init_loop_j_end
        
        i = i + 1
        goto init_loop_i
    label init_loop_i_end
    
    # Perform Matmul
    # C[i,j] += A[i,k] * B[k,j]
    
    i = 0
    label loop_i
    if i == N goto loop_i_end
    
        j = 0
        label loop_j
        if j == N goto loop_j_end
        
            # Offset C (Element Index)
            c_offset = i * N
            c_offset = c_offset + j
            
            # Base Offset C = 2 * elems
            c_base_offset = elems_per_matrix * 2
            c_final_offset = c_offset + c_base_offset
            
            sum = 0
            
            k = 0
            label loop_k
            if k == N goto loop_k_end
            
                # A_offset = i * N + k
                a_offset = i * N
                a_offset = a_offset + k
                
                # B_offset = k * N + j + elems
                b_offset = k * N
                b_offset = b_offset + j
                b_offset = b_offset + elems_per_matrix
                
                val_a = mem[a_offset]
                val_b = mem[b_offset]
                
                prod = val_a * val_b
                sum = sum + prod
                
                k = k + 1
                goto loop_k
            label loop_k_end
            
            mem[c_final_offset] = sum
            
            j = j + 1
            goto loop_j
        label loop_j_end
        
        i = i + 1
        goto loop_i
    label loop_i_end
    
    dummy = mem[0]
    free(mem)
    return dummy
}
