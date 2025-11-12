import pyvista as pv
import numpy as np


nx, ny, nz = 200, 200, 100
x = np.linspace(0, 2000, nx)
y = np.linspace(0, 1600, ny)
z = np.linspace(0, 1000, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

grid = pv.StructuredGrid(X, Y, Z)

values = np.sin(X / 500) + np.cos(Y / 350) + (Z / 180)
grid["values"] = values.flatten()


plotter = pv.Plotter(notebook=False, shape=(1, 1))  


plotter.add_mesh(grid, scalars="values", cmap="terrain")


plotter.show()


modif_2('x')  
modif_2('y') 
modif_2('z')  


plotter.show()
