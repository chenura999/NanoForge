// Variable Load Test for Contextual Bandit Learning
// Tests the AI's ability to learn decision boundaries
// by varying input sizes from tiny to huge

// Initialize sum
sum = 0

// Test with current input (passed as argument)
i = 0
loop_start:
    if i >= n goto loop_end
    sum = sum + i
    i = i + 1
    goto loop_start
loop_end:

return sum
