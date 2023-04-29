import subprocess, os, pdb

for i in range(10):
    print '\n'

print '*****START OF DIFFERENCE CHECK*****\n'
for output_file in os.listdir('output'):
    if output_file.endswith('.txt'):
        test_number = output_file[:6]
        #pdb.set_trace()
        for expected_output_file in os.listdir('expected_output'):
            if test_number == expected_output_file[:6]:
                print '\n----------\n', test_number, '\n----------\n'
                subprocess.call(['diff', "expected_output/" + expected_output_file, "output/" + output_file])


for i in range(10):
    print '\n'
