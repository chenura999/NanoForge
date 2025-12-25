fn main() {
    size = 1000
    ptr = alloc(size)
    
    val = 100
    index = 0
    ptr[index] = val
    
    res = ptr[index]
    
    mul_res = res * 2
    
    free(ptr)
    
    return mul_res
}
