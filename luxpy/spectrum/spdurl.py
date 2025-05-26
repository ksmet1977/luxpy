# spdurl.py

"""
Code SPD data to a URL / DeCode SPDURL to a SPD.

Python port from: https://github.com/herf/spdurl/blob/master/spdurl.js

Translated by ChatGPT-4o (25/05/2025)
"""

import math
import urllib.parse
import numpy as np

__all__ = ['SPD','encodeSPD', 'decodeSPD', 'UnitMap']

# Constants
version = "spd1"
expbase = math.sqrt(math.sqrt(2))
bitscale = 4095
ibitscale = 1.0 / bitscale
gamma = 2.0
b64enc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
b64dec = {ch: i for i, ch in enumerate(b64enc)}

# Unit mapping
UnitMap = {
    "as": "Action spectrum (power)",
    "tr": "Fraction transmitted", 
    "re": "Fraction reflected",
    "uw": "uW total",
    "mw": "mW total", 
    "w": "W total", 
    "uwi": "uW/cm^2", 
    "mwi": "mW/cm^2", 
    "wi": "W/m^2",
    "uwr": "uW/cm^2/sr", 
    "mwr": "mW/cm^2/sr", 
    "wr": "W/m^2/sr",
    "uj": "uJ total", 
    "mj": "mJ total", 
    "j": "J total", 
    "uji": "uJ/cm^2", 
    "mji": "mJ/cm^2",
    "ji": "J/m^2",
    "ujr": "uJ/cm^2/sr", 
    "mjr": "mJ/cm^2/sr", 
    "jr": "J/m^2/sr",
    "q": "quanta (photons)",
    "qi": "q/cm^2", 
    "qr": "q/cm^2/sr",
    "lq": "log10(quanta)", 
    "lqi": "log10(quanta)/cm^2", 
    "lqr": "log10(quanta)/cm^2/sr",
    "mm": "mmol",
    "mmi": "mmol/cm^2",
    "mmr": "mmol/cm^2/sr"
}


class SPD:
    def __init__(self):
        self.base = 380
        self.delta = 1
        self.data = np.array([], dtype=np.float64)
        self.unit = "uwi"
        self.name = None
        self.date = 0
        self.loc = []

    def err(self, spd, linear=False):
        if self.base != spd.base or self.delta != spd.delta or self.data.size != spd.data.size:
            return -1
        valid = self.data > 0
        if linear:
            avg = np.mean(spd.data)
            rel_diff = (spd.data[valid] - self.data[valid]) / avg
        else:
            rel_diff = (spd.data[valid] - self.data[valid]) / self.data[valid]
        return np.sqrt(np.mean(rel_diff ** 2))

    def Unit(self):
        return UnitMap.get(self.unit)


def encodeSPD(spd: SPD) -> str:
    result = [version, str(spd.base), str(spd.delta), spd.unit]

    maxval = np.max(spd.data)
    shexp = math.ceil(math.log(maxval) / math.log(expbase))
    maxval = math.pow(expbase, shexp)
    result.append(str(shexp))

    # Normalize and gamma-encode
    normed = np.clip(spd.data / maxval, 0, 1)
    encoded = np.round(bitscale * np.power(normed, 1.0 / gamma)).astype(np.uint16)

    # Base64 encode
    chars = []
    for val in encoded:
        f1 = (val >> 6) & 63
        f2 = val & 63
        chars.append(b64enc[f1])
        chars.append(b64enc[f2])
    result.append("".join(chars))

    if spd.date:
        result.append(f"d{spd.date}")
    if spd.name:
        result.append(f"n{urllib.parse.quote(spd.name)}")
    if spd.loc and len(spd.loc) == 2:
        result.append(f"l{spd.loc[0]}:{spd.loc[1]}")

    return ",".join(result)


def decodeSPD(s: str) -> SPD:
    tok = s.split(',')
    if len(tok) < 5 or tok[0] != version:
        return None

    spd = SPD()
    spd.base = float(tok[1])
    spd.delta = float(tok[2])
    spd.unit = tok[3]
    shexp = int(tok[4])
    b64 = tok[5]
    shbase = math.pow(expbase, shexp)

    # Decode base64 to array
    vals = np.array([
        (b64dec[b64[i]] << 6) + b64dec[b64[i + 1]]
        for i in range(0, len(b64), 2)
    ], dtype=np.float64)

    vals *= ibitscale
    spd.data = shbase * np.power(vals, gamma)

    for ti in tok[6:]:
        key = ti[0]
        val = ti[1:]
        if key == 'd':
            spd.date = int(val)
        elif key == 'n':
            spd.name = urllib.parse.unquote(val)
        elif key == 'l':
            parts = val.split(':')
            if len(parts) == 2:
                spd.loc = [float(parts[0]), float(parts[1])]

    return spd
    
if __name__ == '__main__':
	
    import numpy as np
	

    wavelengths = np.arange(300, 831, 5)
    data = np.exp(-0.5 * ((wavelengths - 560) / 50) ** 2)

    spd = SPD()
    spd.base = 300
    spd.delta = 5
    spd.data = data
    spd.unit = "wr"
    spd.name = "test spd"
    spd.date = 20250525
    spd.loc = [52.37, 4.89]

    encoded = encodeSPD(spd)
    print("Encoded SPD:")
    print(encoded)

    decoded_spd = decodeSPD(encoded)
    print("\nDecoded SPD metadata:")
    print(f"Name: {decoded_spd.name}")
    print(f"Base: {decoded_spd.base}, Delta: {decoded_spd.delta}, Unit: {decoded_spd.unit}")
    print(f"Location: {decoded_spd.loc}, Date: {decoded_spd.date}")
    print(f"Data sample (first 10): {decoded_spd.data[:10]}")

    err = spd.err(decoded_spd)
    print(f"\nReconstruction RMSE (relative): {err:.6f}")


