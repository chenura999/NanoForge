fn main() {
    val = fib(10)
    return val
}

fn fib(n) {
    if n < 2 goto base_case
    
    a_in = n - 1
    n_save = n
    a = fib(a_in)
    
    b_in = n_save - 2
    b = fib(b_in)
    
    sum = a + b
    return sum
    
    label base_case
    return n
}
