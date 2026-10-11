import numpy as np
import luxpy as lx

xyz = lx.cct_to_xyz(np.array([[20000.0], [13000.0], [12000.0],[19000.0]]), duv=np.zeros((4, 1)), cieobs='1931_2')

print(lx.xyz_to_cct(xyz, mode='zhang2019')[:, 0])           # [20000.  11066.2]
print(lx.xyz_to_cct(xyz[::-1], mode='zhang2019')[::-1, 0])  # [20000.  13000. ]
print(lx.xyz_to_cct(xyz[1:], mode='zhang2019')[:, 0])       # [13000.]