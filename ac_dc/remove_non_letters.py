from tqdm import tqdm

output = open("outputs/test_non_alpha_removed.txt", "w")

def is_allowed(c):
    if c.isalpha():
        return True
    if c == " " or c == "\t" or c == "\n":
        return True
    if c == "." or c == ",":
        return True
    return False

with open("outputs/test_no_lists.txt", "r") as file:
    for line in tqdm(file, total=1344075912):
        output.write("".join(c for c in line if is_allowed(c)))

output.close()