fn main() {
    # Test Loop
    sum = 0
    i = 0
    
    loop_start:
    if i == 10 goto loop_end
        sum = sum + i
        i = i + 1
    goto loop_start
    
    loop_end:
    # Sum 0..9 = 45
    if sum != 45 goto fail
    
    # Test Function Call
    ret = helper(10, 20)
    if ret != 30 goto fail
    
    return 0
    
    label fail
    return 1
}

fn helper(a, b) {
    res = a + b
    return res
}
