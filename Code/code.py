import numpy as np
import pyvista as pv
from noise import pnoise3

vectorized_pnoise3 = np.vectorize(pnoise3)

nx, ny, nz = 200, 200, 80
x = np.linspace(0, 2000, nx)
y = np.linspace(0, 1600, ny)
z = np.linspace(0, 1000, nz)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

def geological_layers(X, Y, Z, fold_intensity=1.0):
    base_layers = np.sin(X / 500) + np.cos(Y / 350) + (Z / 180)
    sedimentation = np.sin(Z / 90) * 0.6 + np.cos(X / 700) * 0.3
    reliefs = 0.2 * np.sin(X / 200) * np.cos(Y / 300)
    noise = 0.4 * vectorized_pnoise3(X / 400, Y / 400, Z / 400)
    folds = fold_intensity * np.sin(X / 400) * np.cos(Y / 300) * 0.5
    return base_layers + sedimentation + reliefs + noise + folds

def add_faults(X, Y, Z, values, fault_intensity=1.0):
    fault1 = np.where(X > 900, (Z - (X - 900) / 2.5) / 45, 0)
    fault2 = np.where(Y > 1100, (Z - (Y - 1100) / 3) / 55, 0)
    shear_zone = np.exp(-((X - 1200) ** 2 + (Y - 900) ** 2) / 300000) * 0.8
    return values + fault_intensity * (fault1 + fault2 + shear_zone)

plotter = pv.Plotter(notebook=False, shape=(1, 3))

def modif_1(fault_intensity=1.0, fold_intensity=1.0):
    values = geological_layers(X, Y, Z, fold_intensity)
    values_with_faults = add_faults(X, Y, Z, values, fault_intensity)
    grid["values"] = values_with_faults.ravel()
    contours = grid.contour(isosurfaces=30, scalars="values")
    plotter.subplot(0, 0)
    plotter.remove_actor('contours')
    plotter.add_mesh(contours, name='contours', opacity=0.5, cmap="viridis")
    modif_2()

def modif_2(normal='x'):
    plotter.subplot(0, 1)
    plotter.remove_actor('slice')
    slice_actor = grid.slice(normal=[1, 0, 0] if normal == 'x' else [0, 1, 0] if normal == 'y' else [0, 0, 1])
    plotter.add_mesh(slice_actor, name='slice', cmap="magma", opacity=0.7)

values = geological_layers(X, Y, Z)
values_with_faults = add_faults(X, Y, Z, values)

grid = pv.StructuredGrid(X, Y, Z)
grid["values"] = values_with_faults.ravel()
contours = grid.contour(isosurfaces=30, scalars="values")
fault_map = np.abs(add_faults(X, Y, Z, np.zeros_like(values)))
grid["fault_intensity"] = fault_map.ravel()
fault_mesh = grid.contour(isosurfaces=10, scalars="fault_intensity")

plotter.set_background("white")
plotter.add_slider_widget(lambda value: modif_1(fault_intensity=value), rng=[0, 2], value=1, title="Fault Intensity", pointa=(0.02, 0.9), pointb=(0.3, 0.9))
plotter.add_slider_widget(lambda value: modif_1(fold_intensity=value), rng=[0, 2], value=1, title="Fold Intensity", pointa=(0.35, 0.9), pointb=(0.65, 0.9))

well_locations = np.array([
    [1000, 800, 500],
    [1500, 600, 400],
    [700, 1200, 300],
])
well_labels = [f"Well {chr(65+i)}" for i in range(len(well_locations))]
cylinder_radius = 30

for loc, label in zip(well_locations, well_labels):
    cylinder = pv.Cylinder(center=loc, direction=(0, 0, -1), radius=cylinder_radius, height=500)
    column = pv.Line(pointa=(loc[0], loc[1], 0), pointb=(loc[0], loc[1], loc[2]))
    plotter.subplot(0, 0)
    plotter.add_mesh(cylinder, color="blue", opacity=0.3, label=label)
    plotter.add_mesh(column, color="red", line_width=3, label=f"Column {label}")
    plotter.subplot(0, 2)
    plotter.add_mesh(cylinder, color="blue", opacity=0.3, label=label)
    plotter.add_mesh(column, color="red", line_width=3, label=f"Column {label}")

plotter.subplot(0, 0)
plotter.add_point_labels(well_locations, labels=well_labels, point_size=12, font_size=22, text_color="black")
plotter.add_mesh(contours, name='contours', opacity=0.5, cmap="viridis", label="Geological Isosurfaces")
plotter.add_mesh(fault_mesh, name='fault_mesh', cmap="coolwarm", opacity=0.6, label="Geological Faults")
plotter.enable_eye_dome_lighting()

plotter.subplot(0, 2)
plotter.add_axes(line_width=2, color="black")
plotter.show_bounds(grid="front", location="outer", color="black")
plotter.add_legend()
plotter.add_text("Wells Only", position="upper_edge", font_size=14, color="black")

plotter.show(title="Interactive Geological Model with Faults and Wells")
