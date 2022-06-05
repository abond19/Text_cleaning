import re
from tqdm import tqdm
new_file = open('outputs/test_no_links.txt', 'w')
with open("outputs/test_non_alpha_removed.txt", "r") as f:
    for line in tqdm(f, total=1344075912):
        text = re.sub(r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?',"",line)
        new_file.write(text)

new_file.close()   