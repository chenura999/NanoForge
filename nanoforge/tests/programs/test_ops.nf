fn main() {
    # Test ops: + - *
    a = 10
    b = 20
    c = a + b
    if c != 30 goto fail
    
    d = b - a
    if d != 10 goto fail
    
    e = a * b
    if e != 200 goto fail
    
    # Test immediate src1
    f = 100
    g = f - 50
    if g != 50 goto fail
    
    return 0

    label fail
    return 1
}
