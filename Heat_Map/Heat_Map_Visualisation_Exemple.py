import pyvista as pv
import numpy as np

#Increase resolution for more details
nx, ny, nz = 50, 50, 30 # Number of points in the x, y, z direction
#Generation of values ​​for each axis
x = np.linspace(0, 2000, nx)
y = np.linspace(0, 1600, ny)
z = np.linspace(0, 1000, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
grid = pv.StructuredGrid(X, Y, Z)

#Add complex faults with affected zones
def add_faults(X, Y, Z, values, fault_intensity=1.0): #Factor to adjust the effect of faults
    fault1 = np.where(X > 900, (Z - (X - 900) / 2.5) / 45, 0)
    fault2 = np.where(Y > 1100, (Z - (Y - 1100) / 3) / 55, 0)
    shear_zone = np.exp(-((X - 1200) ** 2 + (Y - 900) ** 2) / 300000) * 0.8
    return values + fault_intensity * (fault1 + fault2 + shear_zone)

values = np.zeros_like(X)
fault_intensity = 1.0
fault_map = np.abs(add_faults(X, Y, Z, np.zeros_like(values), fault_intensity))
grid["fault_intensity"] = fault_map.ravel()
fault_mesh = grid.contour(isosurfaces=10, scalars="fault_intensity")
plotter = pv.Plotter()
plotter.add_mesh(fault_mesh, name='fault_mesh', cmap="Spectral", opacity=0.5)
plotter.show()
