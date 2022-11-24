output_file = open("samples/sketchengine_sample.txt", "w")

with open("outputs/sketchengine_finished.txt") as f:
    for i in range(1001):
        line = f.readline()
        output_file.write(line)

output_file.close()
        