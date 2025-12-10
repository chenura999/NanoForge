fn fib(n) {
  limit = 2
  if n < limit goto base
  
  one = 1
  two = 2
  
  # fib(n-1)
  n1 = n - one
  a = fib(n1)
  
  # fib(n-2)
  n2 = n - two
  b = fib(n2)
  
  res = a + b
  return res

base:
  return n
}

fn main() {
  arg = 10
  res = fib(arg)
  return res
}
