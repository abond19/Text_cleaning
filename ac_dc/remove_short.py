import os
import re
from tqdm import tqdm

from multiprocessing import Pool, cpu_count

def check_num_proc(num_proc: int = -1) -> int:
    """
    Check the number of processors. Return a safe-checked value.

    Parameters
    ----------
    num_proc : int, optional
        Number of processors to use, by default -1

    Returns
    -------
    int
        Number of processors to use

    Raises
    ------
    ValueError
        If the input exceeds the number of processors available
    """
    maximum: int = cpu_count()
    if num_proc > maximum:
        raise ValueError(
            f"{num_proc} exceeds the maximum number ({maximum}) of processors"
        )

    if num_proc == -1:
        num_proc = maximum
    else:
        print(f"Using {num_proc} processors out of {maximum} can be slow")

    return num_proc


if __name__ == "__main__":
    output_file = open("outputs/removed_short_2.txt", "w")
    
    pool = Pool(check_num_proc())
    
    with open("outputs/removed_short.txt") as f:
        for line in tqdm(f, total=804565159):
        #for line in tqdm(f, total=1344075912):
            #cleaned = clean_sentence(line)
            if len(line) > 100:
                output_file.write(line)

    output_file.close()
    