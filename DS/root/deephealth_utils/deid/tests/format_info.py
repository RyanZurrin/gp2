with open('streets.txt', 'r') as streets:
    with open('formatted_streets.txt', 'w') as output:
        for next_line in streets:
            if next_line != '\n':
                next_line = next_line.splitlines()[0]
                output.write(', \"' + next_line + '\"')
