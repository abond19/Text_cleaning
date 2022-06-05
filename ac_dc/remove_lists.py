import re
from tqdm import tqdm

output_file = open("outputs/test_no_lists.txt", "w")

pattern = "\|[ ]*[\-\-]*[\w ışöüğÜ'\-çİÖ©]+ \|"
with open("outputs/test.csv") as f:
    for line in tqdm(f, total=1344075912):
        match = re.findall(pattern, line, re.DOTALL)
        if match is not []:
            output_file.write(re.sub(pattern, "", line, re.DOTALL))
        else:
            output_file.write(line)

output_file.close()
