fn main() {
    val = fib(10)
    return val
}

fn fib(n) {
    if n == 0 goto return_zero
    
    a = 0
    b = 1
    i = 1
    
    label loop
    if i == n goto return_b
    
    temp = a + b
    a = b
    b = temp
    i = i + 1
    goto loop
    
    label return_b
    return b

    label return_zero
    return 0
}
