sum = 0
i = 5
label loop
if i == 0 goto end
sum = sum + i
i = i - 1
goto loop
label end
return sum
