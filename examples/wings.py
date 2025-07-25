import os
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from HDM import hdm_embed, HDMConfig

directory_path = 'example-data/wing/'
files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
files = files[:50]
data_samples = [np.loadtxt(os.path.join(directory_path, file), delimiter=',') for file in files]

sample_length = len(data_samples[0])
# print(sample_length)
num_samples = len(data_samples)
# num_samples = 4

config = HDMConfig(
    base_epsilon = 0.004,
    fiber_epsilon = 0.0006,
    base_sparsity = 0.08,
    base_knn = None,
    fiber_sparsity = 0.08,
    fiber_knn = None,

)

diffusion_coords = hdm_embed(
    data_samples = data_samples,
    config = config
)
# diffusion_coords = diffusion_coords[:, :3]  
# normalize each vector 
# for i in range(diffusion_coords.shape[0]//110):
#     N = np.sqrt(np.sum(diffusion_coords[i*110:(i+1)*110, :3]**2, axis=1)).sum()
#     diffusion_coords[i*110:(i+1)*110, :3] /= N
    

# point_cloud = pv.PolyData(diffusion_coords[:10, :3])
# plotter = pv.Plotter()

# scalars = np.tile(np.arange(sample_length), num_samples)
# cmap = plt.get_cmap("rainbow", sample_length)
# norm = Normalize(vmin=0, vmax=sample_length-1)

# plotter.add_mesh(point_cloud, scalars=scalars[:10], point_size=10, 
#                  render_points_as_spheres=True, cmap="tab20", 
#                  clim=[0, sample_length-1], show_scalar_bar=False)

# plotter.show()


tab20 = plt.colormaps.get_cmap('tab20').resampled(20)
set1 = plt.colormaps.get_cmap('Set1').resampled(9)
set2 = plt.colormaps.get_cmap('Set2').resampled(8)
set3 = plt.colormaps.get_cmap('Set3').resampled(12)
pastel1 = plt.colormaps.get_cmap('Pastel1').resampled(9)
pastel2 = plt.colormaps.get_cmap('Pastel2').resampled(8)
dark2 = plt.colormaps.get_cmap('Dark2').resampled(8)
accent = plt.colormaps.get_cmap('Accent').resampled(8)

# Extract colors and combine
all_colors = []
for cmap, n in [(tab20, 20), (set1, 9), (set2, 8), (set3, 12), 
                (pastel1, 9), (pastel2, 8), (dark2, 8), (accent, 8)]:
    all_colors.extend([cmap(i) for i in range(n)])
colors_110 = all_colors[:110]
combined_cmap = ListedColormap(colors_110)

wings = [0,1,2,3]
embedded_wings = len(wings)
wing_size = 110
enable_labels = False


# Create a plotter with 2 subplots (1 row, 2 columns)
plotter1 = pv.Plotter(shape=(1,1))
plotter2 = pv.Plotter(shape=(2,2))

right_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
plotter2_text = ["circle", "square", "cylinder", "cone"]
type2_scale = 0.001
type1_scale = 0.00001

shape_type2 = [pv.Sphere(radius=type2_scale), pv.Cube(x_length=type2_scale, y_length=type2_scale, z_length=type2_scale), pv.Cylinder(radius=type2_scale, height=2*type2_scale), pv.Cone(radius=type2_scale, height=2*type2_scale)]
shape_type1 = [pv.Sphere(radius=type1_scale), pv.Cube(x_length=type1_scale, y_length=type1_scale, z_length=type1_scale), pv.Cylinder(radius=type1_scale, height=2*type1_scale), pv.Cone(radius=type1_scale, height=2*type1_scale)]

scalars1 = np.tile(np.arange(sample_length), num_samples)
for i in wings:
    start_idx = i * wing_size
    end_idx = (i+1) * wing_size
    
    # First subplot
    point_cloud1 = pv.PolyData(diffusion_coords[start_idx:end_idx, :3])
    
    # Add scalars to the original point cloud BEFORE glyphing
    point_cloud1['colors'] = scalars1[:wing_size]
    
    # Create glyphed mesh
    glyphed1 = point_cloud1.glyph(geom=shape_type1[i], scale=False)
    
    # Add the glyphed mesh (remove point_size and render_points_as_spheres)
    plotter1.add_mesh(glyphed1, cmap=combined_cmap, 
                     clim=[0, sample_length-1], show_scalar_bar=False)

    # Add labels for each point (use original point cloud for labels)
    if enable_labels:
        labels1 = [str(j) for j in range(wing_size)]  # Changed i to j to avoid confusion
        plotter1.add_point_labels(
            point_cloud1,  
            labels=labels1,
            point_size=20, 
            font_size=12,
            always_visible=True,
            text_color='black',
            fill_shape=False
        )
    plotter1.add_text(f"Embedded wings", position='upper_left')

    # Second subplot
    plotter2.subplot(right_positions[i][0], right_positions[i][1])

    points_2d = data_samples[i]
    points_3d = np.column_stack([points_2d, np.zeros(len(points_2d))])
    point_cloud2 = pv.PolyData(points_3d)
    
    # Add scalars to the original point cloud BEFORE glyphing
    scalars2 = np.tile(np.arange(sample_length), num_samples)
    point_cloud2['colors'] = scalars2[:wing_size]
    
    # Create glyphed mesh
    glyphed2 = point_cloud2.glyph(geom=shape_type2[i], scale=False)
    
    # Add the glyphed mesh (not the original point_cloud2)
    plotter2.add_mesh(glyphed2, cmap=combined_cmap, 
                     clim=[0, sample_length-1], show_scalar_bar=False)

    # Add labels for each point (use original point cloud for labels)
    if enable_labels:
        labels2 = [str(j) for j in range(wing_size)]  # Changed i to j
        plotter2.add_point_labels(point_cloud2, 
            labels2, 
            point_size=20, 
            font_size=12,
            always_visible=True,
            text_color='black',
            fill_shape=False
        )

    plotter2.add_text(f"{plotter2_text[i]}", position='upper_left')


plotter1.show(interactive=False, auto_close=False)
plotter2.show()
pv.close_all()
del plotter1
del plotter2