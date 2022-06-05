output_file = open("samples/end_sample.txt", "w")

with open("outputs/removed_short_2.txt") as f:
    for i in range(1001):
        line = f.readline()
        output_file.write(line)

output_file.close()
        