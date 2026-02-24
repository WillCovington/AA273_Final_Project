import re
import numpy as np
import requests
from urllib.parse import urljoin

# this script helps to average out all of the clones from the GRAIL mission, which compiled data on all of the moon's spherical harmonics
# NOTE! This process takes a loooooong while since we have to download the data for all 500 clones, so ideally we'd only run this a couple of times for sizes like 50, 100, 150, 200, 250, ..., and then store the results for later use. 

BASE_DIRS = [
    "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/extras/clones/gggrx_1200a_clones_0001_0100/",
    "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/extras/clones/gggrx_1200a_clones_0101_0200/",
    "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/extras/clones/gggrx_1200a_clones_0201_0300/",
    "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/extras/clones/gggrx_1200a_clones_0301_0400/",
    "https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/extras/clones/gggrx_1200a_clones_0401_0500/",
]

# Match the coefficient lines: "n, m, C, S" (commas + whitespace)
LINE_RE = re.compile(
    r"^\s*(\d+)\s*,\s*(\d+)\s*,\s*([+-]?\d+\.\d+e[+-]?\d+)\s*,\s*([+-]?\d+\.\d+e[+-]?\d+)\s*$",
    re.IGNORECASE,
)

# this just puts the URL together based on which 1200a clone we want to use
def clone_url(k: int) -> str:
    # k from 1..500
    if not (1 <= k <= 500):
        raise ValueError("k must be in 1..500")
    if k <= 100:
        base = BASE_DIRS[0]
    elif k <= 200:
        base = BASE_DIRS[1]
    elif k <= 300:
        base = BASE_DIRS[2]
    elif k <= 400:
        base = BASE_DIRS[3]
    else:
        base = BASE_DIRS[4]
    fname = f"gggrx_1200a_clone{k:04d}_sha.tab"
    return urljoin(base, fname)

def stream_clone_coeffs(url: str, L: int):
    # returns radius in Km, G * M in km^3/s^2, and the c and s coefficients

    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    it = r.iter_lines(decode_unicode=True)

    # parse the file header
    header = next(it).strip()
    parts = [p.strip() for p in header.split(",")]

    R_km = float(parts[0])
    GM_km3_s2 = float(parts[1])

    def coeff_generator():
        for line in it:
            if not line:
                continue
            m = LINE_RE.match(line)
            if not m:
                continue
            n = int(m.group(1))
            if n > L:
                break
            mm = int(m.group(2))
            if mm > n:
                continue
            c = float(m.group(3))
            s = float(m.group(4))
            yield n, mm, c, s

    return R_km, GM_km3_s2, coeff_generator()

def average_clones(L: int, k_start: int = 1, k_end: int = 500):
    # Running mean 
    C_mean = np.zeros((L + 1, L + 1), dtype=np.float64)
    S_mean = np.zeros((L + 1, L + 1), dtype=np.float64)
    R_km_ref = None
    GM_km3_s2_ref = None

    count = 0

    for k in range(k_start, k_end + 1):
        url = clone_url(k)

        R_km, GM_km3_s2, coeffs = stream_clone_coeffs(url, L)

        # we take R_km and GM_km3_s2 from the first clone only
        if count == 0: 
            R_km_ref = R_km
            GM_km3_s2_ref = GM_km3_s2

        count += 1
        # initialize our Ck and Sk matrices for this specific clone
        Ck = np.zeros((L + 1, L + 1), dtype=np.float64)
        Sk = np.zeros((L + 1, L + 1), dtype=np.float64)

        for n, m, c, s in coeffs:
            Ck[n, m] = c
            Sk[n, m] = s

        # Update running mean
        C_mean += (Ck - C_mean) / count
        S_mean += (Sk - S_mean) / count

        print(f"Processed clone {k:04d} ({count} total)")

    return C_mean, S_mean, R_km_ref, GM_km3_s2_ref

if __name__ == "__main__":
    L = 660  # this is the truncation degree (basically what the final size our matrix is going to be)
    Cbar, Sbar = average_clones(L)
    np.savez(f"grgm1200a_clone_mean_L{L}.npz", C=Cbar, S=Sbar, R_km=R_km, GM_km3_s2=GM_km3_s2)
    print("Saved mean coefficients.")
