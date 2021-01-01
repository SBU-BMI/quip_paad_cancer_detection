import os
import sys
input = sys.argv[1]
output = sys.argv[2]

f1 = open(input, 'r')
f2 = open('temp.txt', 'w')
start = 0
for line in f1:
    parts = line.split()
    f2.write(parts[0] + ' ' + parts[1] + ' ' + str(1 - float(parts[2])) + '\n')

f1.close()
f2.close()
command = 'cp temp.txt' + ' ' + output
print('command: ', command)
os.system(command)
os.system('rm -rf temp.txt')
