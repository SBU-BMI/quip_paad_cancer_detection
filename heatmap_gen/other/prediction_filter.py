import os
import sys
input = sys.argv[1]
output = sys.argv[2]

f1 = open(input, 'r')
f2 = open('temp.txt', 'w')
start = 0
for line in f1:
    start += 1
    if start < 3:
        f2.write(line)
        continue
    parts = line.split()
    if float(parts[2]) > 1e-3:
        f2.write(line)

f1.close()
f2.close()
command = 'cp temp.txt' + ' ' + output
print('command: ', command)
os.system(command)
os.system('rm -rf temp.txt')
