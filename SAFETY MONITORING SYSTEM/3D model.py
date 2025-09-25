import os
import tempfile
import time
import numpy as np
import gradio as gr
import trimesh
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter, sobel
from gradio_litmodel3d import LitModel3D
import traceback
import shutil
import cv2
from skimage import feature, color, transform, exposure
import matplotlib.pyplot as plt

class EnhancedModelGenerator:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_models")
        os.makedirs(self.output_dir, exist_ok=True)
        self.last_model_path = None   
    def barycentric_coords(self, p, a, b, c):
        """Calculate barycentric coordinates of point p in triangle abc"""
        v0 = b - a
        v1 = c - a
        v2 = p - a
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return np.array([1.0, 0.0, 0.0])
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return np.array([u, v, w])
    def create_heightmap_from_image(self, image, detail_level=1.0, contrast=1.0, edge_weight=0.5, normal_smoothing=1.0):
        """ Create an enhanced height map from image using advanced multi-scale processing """
        if image.mode != 'L':
            gray_image = image.convert('L')
        else:
            gray_image = image
        enhancer = ImageEnhance.Contrast(gray_image)
        gray_image = enhancer.enhance(contrast)
        height_map = np.array(gray_image).astype(float) / 255.0
        if edge_weight > 0:
            img_array = np.array(gray_image)
            edges1 = feature.canny(img_array, sigma=1.5)
            edges2 = feature.canny(img_array, sigma=3)
            sobel_h = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3) / 255.0
            sobel_v = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3) / 255.0
            sobel_mag = np.sqrt(sobel_h**2 + sobel_v**2)
            sobel_mag = sobel_mag / sobel_mag.max()  
            combined_edges = (edges1.astype(float) * 0.5 + 
                             edges2.astype(float) * 0.3 + 
                             sobel_mag * 0.2)
            height_map = (1 - edge_weight) * height_map + edge_weight * combined_edges
        if detail_level != 1.0:
            scales = [1.5, 3.0, 6.0]
            details_sum = np.zeros_like(height_map)
            scale_weights = [0.6, 0.3, 0.1]
            for i, scale in enumerate(scales):
                blurred = gaussian_filter(height_map, sigma=scale)
                detail_layer = height_map - blurred
                details_sum += detail_layer * scale_weights[i] * detail_level
            base_layer = gaussian_filter(height_map, sigma=max(scales))
            height_map = base_layer + details_sum
        height_map_uint8 = (height_map * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        height_map_equalized = clahe.apply(height_map_uint8)
        height_map = height_map * 0.7 + (height_map_equalized / 255.0) * 0.3
        min_val = height_map.min()
        max_val = height_map.max()
        if max_val > min_val: 
            height_map = (height_map - min_val) / (max_val - min_val)
        sobel_x = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=5)
        normal_map = np.zeros((height_map.shape[0], height_map.shape[1], 3))
        normal_map[..., 0] = -sobel_x
        normal_map[..., 1] = -sobel_y
        z_scale = 0.15 / max(np.max(np.abs(sobel_x)), np.max(np.abs(sobel_y)))
        normal_map[..., 2] = z_scale
        norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
        norm[norm == 0] = 1e-8
        normal_map = normal_map / norm
        if normal_smoothing > 0:
            normal_map_01 = (normal_map + 1) / 2
            for i in range(3):
                normal_map_01[..., i] = cv2.bilateralFilter(
                    normal_map_01[..., i].astype(np.float32), 
                    d=int(normal_smoothing * 5),  
                    sigmaColor=0.1,              
                    sigmaSpace=normal_smoothing  
                )
        
        # Convert back to -1 to 1 range
            normal_map = normal_map_01 * 2 - 1
        
            # Renormalize after smoothing
            norm = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
            norm[norm == 0] = 1e-8
            normal_map = normal_map / norm
    
    # Convert to 0-1 range for easier use later
        normal_map = (normal_map + 1) / 2
    
        return height_map, normal_map

    
    
    def generate_ambient_occlusion(self, mesh, samples=64):
        """
        Generate ambient occlusion with improved sampling for more realistic lighting
        """
        try:
            # Use hemisphere sampling instead of sphere sampling for more realistic AO
            # (hemisphere aligned with surface normal is more physically accurate)
            hemisphere_dirs = []
            for _ in range(samples):
                # Random point on unit hemisphere (z >= 0)
                theta = 2 * np.pi * np.random.random()
                phi = np.arccos(np.random.random())  # Cosine-weighted distribution
                x = np.sin(phi) * np.cos(theta)
                y = np.sin(phi) * np.sin(theta)
                z = np.cos(phi)
                hemisphere_dirs.append([x, y, z])
        
            ray_directions = np.array(hemisphere_dirs)
        
        # Calculate ambient occlusion with hemisphere aligned to normals
            hits = np.zeros((len(mesh.vertices), samples), dtype=bool)
        
        # Create a surface-aligned ray for each vertex
            for i, vertex_normal in enumerate(mesh.vertex_normals):
                # Align hemisphere with vertex normal using rotation matrix
                up = np.array([0, 0, 1])
                if np.allclose(vertex_normal, up) or np.allclose(vertex_normal, -up):
                    # Special case for vertical normals
                    right = np.array([1, 0, 0])
                    forward = np.cross(up, right)
                else:
                    right = np.cross(up, vertex_normal)
                    right = right / np.linalg.norm(right)
                    forward = np.cross(vertex_normal, right)
            
                # Local to world transformation matrix
                rotation = np.column_stack((right, forward, vertex_normal))
            
                # Transform hemisphere rays to align with normal
                aligned_dirs = np.array([rotation @ dir for dir in ray_directions])
            
                # Calculate origin with small offset
                ray_origin = mesh.vertices[i] + vertex_normal * 1e-4
            
                # Cast rays in hemisphere directions
                for j, direction in enumerate(aligned_dirs):
                    try:
                        locations, _, _ = mesh.ray.intersects_location(
                            ray_origins=[ray_origin],
                            ray_directions=[direction],
                            multiple_hits=False
                        )
                        if len(locations) > 0:
                            hits[i, j] = True
                    except Exception:
                        continue
        
        # Apply distance-based weighting (closer hits contribute more to occlusion)
        # This is a simplified approximation
            vertex_ao = 1.0 - np.mean(hits, axis=1)
        
        # Apply gamma correction to AO for more natural look
            gamma = 1.5  # Higher gamma = darker shadows, lower gamma = lighter shadows
            vertex_ao = np.power(vertex_ao, 1.0/gamma)
        
            return vertex_ao
        except Exception as e:
            print(f"Ambient occlusion generation failed: {e}")
            return np.ones(len(mesh.vertices))

    def adaptive_subdivision(self, vertices, faces, height_map, max_faces=50000, threshold=0.02):
        """
        Adaptively subdivide the mesh in areas with high detail
        
        Args:
            vertices: Vertex array
            faces: Face array
            height_map: Height map for detail detection
            max_faces: Maximum number of faces to create
            threshold: Detail threshold for subdivision
            
        Returns:
            vertices, faces: Subdivided mesh
        """
        try:
            # Calculate gradient magnitude for detail detection
            gradient_x = np.gradient(height_map, axis=1)
            gradient_y = np.gradient(height_map, axis=0)
            gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Mark areas for subdivision
            height, width = height_map.shape
            detail_mask = gradient_mag > threshold
            
            # Create a new mesh with adaptive subdivision
            new_vertices = list(vertices)
            new_faces = []
            
            face_count = 0
            for face in faces:
                # Check if any vertex of the face is in a high-detail area
                v0, v1, v2 = face
                
                # Convert vertex coordinates to image coordinates
                x0 = int((vertices[v0][0] + 1) / 2 * (width-1))
                y0 = int((1 - (vertices[v0][1] + 1) / 2) * (height-1))
                x1 = int((vertices[v1][0] + 1) / 2 * (width-1))
                y1 = int((1 - (vertices[v1][1] + 1) / 2) * (height-1))
                x2 = int((vertices[v2][0] + 1) / 2 * (width-1))
                y2 = int((1 - (vertices[v2][1] + 1) / 2) * (height-1))
                
                # Ensure coordinates are within bounds
                x0, y0 = np.clip(x0, 0, width-1), np.clip(y0, 0, height-1)
                x1, y1 = np.clip(x1, 0, width-1), np.clip(y1, 0, height-1)
                x2, y2 = np.clip(x2, 0, width-1), np.clip(y2, 0, height-1)
                
                is_detailed = detail_mask[y0, x0] or detail_mask[y1, x1] or detail_mask[y2, x2]
                
                if is_detailed and face_count < max_faces:
                    # Subdivide triangle by adding a vertex at the center
                    # Calculate center vertex position
                    center_x = (vertices[v0][0] + vertices[v1][0] + vertices[v2][0]) / 3
                    center_y = (vertices[v0][1] + vertices[v1][1] + vertices[v2][1]) / 3
                    
                    # Interpolate Z value from the three vertices
                    center_z = (vertices[v0][2] + vertices[v1][2] + vertices[v2][2]) / 3
                    
                    # Add the new vertex
                    v_center = len(new_vertices)
                    new_vertices.append([center_x, center_y, center_z])
                    
                    # Create three new triangles
                    new_faces.append([v0, v1, v_center])
                    new_faces.append([v1, v2, v_center])
                    new_faces.append([v2, v0, v_center])
                    
                    face_count += 3
                else:
                    # Keep original triangle
                    new_faces.append(face)
                    face_count += 1
                
                if face_count >= max_faces:
                    # Add remaining faces as-is
                    for remaining_face in faces[list(faces).index(face)+1:]:
                        if face_count < max_faces:
                            new_faces.append(remaining_face)
                            face_count += 1
                        else:
                            break
                    break
                    
            return np.array(new_vertices), np.array(new_faces)
        except Exception as e:
            print(f"Adaptive subdivision failed: {e}")
            # Return the original mesh if subdivision fails
            return vertices, faces
        
    def create_enhanced_texture_map(self, image, height_map):
        """
        Create an enhanced texture map from the input image and height map

        Args:
        image: Input PIL image
        height_map: Generated height map
    
        Returns:
        Enhanced texture map as numpy array
        """
        try:
        # Convert image to numpy array if it's not already
            if hasattr(image, 'convert'):
                # If image is grayscale, convert to RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    img_array = np.array(image)
            else:
                img_array = image
            
        # Resize to match height map dimensions if needed
            height, width = height_map.shape
            if img_array.shape[0] != height or img_array.shape[1] != width:
            # Resize image using PIL for better quality
                pil_img = Image.fromarray(img_array) if not hasattr(image, 'resize') else image
                pil_img = pil_img.resize((width, height), Image.LANCZOS)
                img_array = np.array(pil_img)
        
        # Enhance texture using height map to add detail
        # Create a combined texture that includes derived information from height map
        
        # Calculate shaded relief (hill shade) for more visual depth in texture
            dx = cv2.Sobel(height_map, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(height_map, cv2.CV_64F, 0, 1, ksize=3)
        
        # Simulate sun from top-left
            angle = 45 * np.pi / 180  # 45 degrees in radians
            hillshade = np.cos(angle) * dx + np.sin(angle) * dy + 0.5
        
        # Normalize hillshade
            hillshade = (hillshade - hillshade.min()) / (hillshade.max() - hillshade.min())
        
        # Convert to 3-channel grayscale if needed
            if len(hillshade.shape) == 2:
                hillshade = np.stack([hillshade] * 3, axis=-1)
        
        # Blend original image with hillshade for enhanced texture
        # Apply soft light blending for more natural look
            def soft_light_blend(base, blend):
                result = np.zeros_like(base, dtype=np.float32)
                # Convert base and blend to float
                base = base.astype(np.float32) / 255.0
                blend = blend.astype(np.float32)
            
            # Apply soft light blend formula
                mask = blend <= 0.5
                result[mask] = 2 * base[mask] * blend[mask] + base[mask]**2 * (1 - 2 * blend[mask])
                result[~mask] = 2 * base[~mask] * (1 - blend[~mask]) + np.sqrt(base[~mask]) * (2 * blend[~mask] - 1)
            
                return (np.clip(result, 0, 1) * 255).astype(np.uint8)
        
            enhanced_texture = soft_light_blend(img_array, hillshade)
        
        # Apply subtle contrast enhancement
            lab = cv2.cvtColor(enhanced_texture, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel for better detail
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
        
        # Merge channels back
            lab = cv2.merge((l, a, b))
            enhanced_texture = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
            return enhanced_texture
        
        except Exception as e:
            print(f"Error creating enhanced texture map: {e}")
        # Return the original image if texture enhancement fails
            if hasattr(image, 'convert'):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                return np.array(image)
            return image

    def generate_3d_mesh(self, image, resolution=150, height_scale=0.5, detail_level=1.0, 
                      edge_weight=0.3, smoothing=1.0, contrast=1.2, adaptive_mesh=True,
                      calculate_ao=True):
        """
        Generate an enhanced 3D mesh with improved realism features
        """
        # ... [existing code for processing the image] ...
    
        # Create an enhanced height map from image
        height_map, normal_map = self.create_heightmap_from_image(
            image, 
            detail_level=detail_level,
            contrast=contrast,
            edge_weight=edge_weight,
            normal_smoothing=smoothing
        )
    
        # ... [existing code for resizing] ...
    
    # Add naturalistic micro-detail noise
    # This creates small variations for more realistic surfaces
        noise_scale = 0.03 * detail_level  # Scale noise with detail level
        noise = np.random.randn(*height_map.shape) * noise_scale
    # Apply gaussian to make noise smoother
        noise = gaussian_filter(noise, sigma=1.0)
    # Add noise to height map
        height_map = height_map + noise
    
    # Post-process height map with small median filter to remove outliers
    # This helps prevent "spikes" in the mesh
        height_map = cv2.medianBlur((height_map * 255).astype(np.uint8), 3) / 255.0
    
    # ... [existing code for creating vertices and faces] ...
        height, width = height_map.shape
        x = np.linspace(-1, 1, width)
        y = np.linspace(1, -1, height)
        xv, yv = np.meshgrid(x, y)
    # Create vertices
        vertices = np.zeros((width * height, 3))
        vertices[:, 0] = xv.flatten()
        vertices[:, 1] = yv.flatten()
    
    # Apply improved height scaling with non-linear mapping for more natural results
    # This creates more natural elevation changes (steeper slopes look more realistic)
        elevation = height_map.flatten()
    # Apply slight gamma correction to height values
        height_gamma = 0.8  # Values < 1 emphasize higher areas
        elevation = np.power(elevation, height_gamma)
        # Apply final height scaling
        vertices[:, 2] = elevation * height_scale
    
    # ... [rest of the function] ...
        faces = []
        for i in range(height - 1):
            for j in range(width - 1):
            # Calculate vertex indices
                v0 = i * width + j
                v1 = i * width + (j + 1)
                v2 = (i + 1) * width + j
                v3 = (i + 1) * width + (j + 1)
        
        # Create two triangular faces
                faces.append([v0, v1, v3])
                faces.append([v0, v3, v2])

# Convert faces to numpy array
        faces = np.array(faces)
        
        if image.mode == 'RGB':
    # Use input image colors
            img_resized = image.resize((width, height))
            img_array = np.array(img_resized)
            colors = img_array.reshape(-1, 3)
        else:
    # Create a colormap from height values
    # Convert heightmap to colors using a colormap (blue to red)
            cm = plt.get_cmap('viridis')
            height_colors = cm(height_map.flatten())[:, 0:3]
            colors = (height_colors * 255).astype(np.uint8)
    
    # Create the mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors
        )
    
    # Calculate ambient occlusion
        if calculate_ao and len(mesh.vertices) < 20000:
            ao_values = self.generate_ambient_occlusion(mesh, samples=32)
        
        # Apply ambient occlusion to vertex colors with improved blending
            ao_values = ao_values.reshape(-1, 1)
        # Modified AO application - preserve some color vibrancy
            ao_min = 0.4  # Higher minimum value preserves more color
            ao_values_adjusted = ao_min + (1.0 - ao_min) * ao_values
        
        # Apply AO with HSV-aware darkening (preserves hue better)
            colors_hsv = np.zeros_like(colors, dtype=float)
            for i in range(len(colors)):
            # Convert RGB to HSV
                r, g, b = colors[i] / 255.0
                hsv = cv2.cvtColor(np.array([[[r, g, b]]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0, 0]
            # Only darken value component with AO
                hsv[2] *= ao_values_adjusted[i, 0]
                colors_hsv[i] = hsv
        
        # Convert back to RGB
            new_colors = np.zeros_like(colors)
            for i in range(len(colors_hsv)):
                h, s, v = colors_hsv[i]
                rgb = cv2.cvtColor(np.array([[[h, s, v]]], dtype=np.float32), cv2.COLOR_HSV2RGB)[0, 0]
                new_colors[i] = (rgb * 255).astype(np.uint8)
        
            mesh.visual.vertex_colors = new_colors
        else:
            mesh.visual.vertex_colors = colors
    
    # Add post-processing to smooth mesh while preserving features
        if smoothing > 0 and len(mesh.vertices) < 50000:  # Only for reasonably sized meshes
            # Laplacian smoothing with feature preservation
            vertices_np = np.array(mesh.vertices)
            faces_np = np.array(mesh.faces)
        
        # Calculate vertex adjacency
            adjacency = {}
            for face in faces_np:
                for i in range(3):
                    v1, v2, v3 = face[i], face[(i+1)%3], face[(i+2)%3]
                    if v1 not in adjacency:
                        adjacency[v1] = []
                    if v2 not in adjacency[v1]:
                        adjacency[v1].append(v2)
                    if v3 not in adjacency[v1]:
                        adjacency[v1].append(v3)
        
        # Calculate feature edges (high curvature)
            feature_verts = set()
            for v_idx, neighbors in adjacency.items():
                if len(neighbors) >= 2:
                    v_normal = mesh.vertex_normals[v_idx]
                    neighbor_normals = [mesh.vertex_normals[n] for n in neighbors]
                
                # Check for high curvature
                    for normal in neighbor_normals:
                        dot_product = np.dot(v_normal, normal)
                        # If normals differ significantly, it's a feature edge
                        if dot_product < 0.7:  # About 45 degrees difference
                            feature_verts.add(v_idx)
                            break
        
        # Apply smoothing with feature preservation
            smoothing_strength = 0.5 * smoothing  # Scale to make slider more intuitive
            iterations = max(1, int(smoothing))
        
            for _ in range(iterations):
                new_vertices = vertices_np.copy()
            
                for v_idx in range(len(vertices_np)):
                    if v_idx in adjacency:
                        neighbors = adjacency[v_idx]
                        if len(neighbors) > 0:
                            # Feature-preserving weight
                            weight = 0.1 if v_idx in feature_verts else smoothing_strength
                        
                        # Calculate centroid of neighbors
                            centroid = np.mean([vertices_np[n] for n in neighbors], axis=0)
                        
                        # Move vertex towards centroid based on weight
                            new_vertices[v_idx] = (1-weight) * vertices_np[v_idx] + weight * centroid
            
                vertices_np = new_vertices
        
        # Update mesh vertices
            mesh.vertices = vertices_np
        
        enhanced_texture = self.create_enhanced_texture_map(image, height_map)
    
        # Generate high-quality texture coordinates
        # Calculate texture coordinates based on normalized positions
        texture_coords = np.zeros((len(vertices), 2))
        texture_coords[:, 0] = (vertices[:, 0] + 1) / 2  # Map from [-1,1] to [0,1]
        texture_coords[:, 1] = (vertices[:, 1] + 1) / 2  # Map from [-1,1] to [0,1]
    
    # Flip Y coordinates to match image orientation
        texture_coords[:, 1] = 1 - texture_coords[:, 1]
    
    # Apply texture mapping
        if enhanced_texture.shape[0] > 0 and enhanced_texture.shape[1] > 0:
            # Convert enhanced texture to appropriate format
            height, width = enhanced_texture.shape[:2]
            texture_colors = np.zeros_like(colors)
        
        # Sample texture colors based on texture coordinates
        for i in range(len(texture_coords)):
            u, v = texture_coords[i]
            x = min(int(u * width), width - 1)
            y = min(int(v * height), height - 1)
            texture_colors[i] = enhanced_texture[y, x]
        
        # Blend vertex colors with texture colors for better results
        alpha = 0.8  # Texture contribution weight
        blended_colors = (texture_colors * alpha + colors * (1 - alpha)).astype(np.uint8)
        colors = blended_colors
    
    # Create the mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors
        )
    
    # Apply PBR materials for more realistic appearance
        mesh = self.apply_pbr_materials(mesh, enhanced_texture, normal_map)
    
        return mesh
    
    def apply_pbr_materials(self, mesh, diffuse_texture, normal_map):
        """
        Apply PBR (Physically Based Rendering) materials to the mesh
    
        Args:
        mesh: The trimesh object
        diffuse_texture: Diffuse color texture as numpy array
        normal_map: Normal map as numpy array
        
        Returns:
            Updated mesh with PBR materials
        """
        try:
        # For trimesh, we can add material properties, but full PBR would require
        # exporting to a format that supports it (like glTF/GLB)
        
        # Handle case when mesh has no texture coordinates
            if not hasattr(mesh, 'visual') or not hasattr(mesh.visual, 'uv'):
                # In this case, we'll just use vertex colors as is
                return mesh
            
        # Create a material with PBR properties
            material = trimesh.visual.material.PBRMaterial(
                name="pbr_material",
                roughnessFactor=0.7,  # Medium roughness
                metallicFactor=0.2,   # Low metallic for most natural surfaces
                baseColorTexture=diffuse_texture
            )
        
        # If we have a normal map, try to use it (trimesh support varies)
            if normal_map is not None and hasattr(material, 'normalTexture'):
            # Convert normal map to the right format
                normal_map_rgb = (normal_map * 255).astype(np.uint8)
                material.normalTexture = normal_map_rgb
            
        # Apply material to mesh
            if hasattr(mesh.visual, 'material'):
                mesh.visual.material = material
            
            return mesh
        
        except Exception as e:
            print(f"Error applying PBR materials: {e}")
        # Return the original mesh if material application fails
            return mesh
    
    def run_model(self, input_image, resolution=150, height_scale=0.5, detail_level=1.0, 
                   edge_weight=0.3, smoothing=1.0, contrast=1.2, 
                   adaptive_mesh=True, calculate_ao=True, format="glb"):
        """
        Process the input image and return a 3D model file path with enhanced realism
        """
        try:
            print("\n--- Starting 3D model generation ---")
            start = time.time()
            
            # Generate enhanced 3D mesh
            print(f"Input image type: {type(input_image)}, mode: {input_image.mode if hasattr(input_image, 'mode') else 'N/A'}")
            mesh = self.generate_3d_mesh(
                input_image, 
                resolution=resolution,
                height_scale=height_scale,
                detail_level=detail_level,
                edge_weight=edge_weight,
                smoothing=smoothing,
                contrast=contrast,
                adaptive_mesh=adaptive_mesh,
                calculate_ao=calculate_ao
            )
            
            print(f"Mesh created with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
            
            # Create a unique filename based on timestamp
            timestamp = int(time.time())
            filename = f"model_{timestamp}.{format}"
            output_path = os.path.join(self.output_dir, filename)
            
            print(f"Exporting mesh to {output_path}")
            export_options = {}

            if format == "glb":
    # GLB format can store PBR materials
                export_options = {}
            elif format == "obj":
                # OBJ format needs separate texture files
                texture_dir = os.path.join(os.path.dirname(output_path), "textures")
                os.makedirs(texture_dir, exist_ok=True)
    
                export_options = {
                    'include_normals': True,
                    'include_texture': True,
                    'mtl_name': os.path.splitext(os.path.basename(output_path))[0] + ".mtl",
                    'write_texture_separately': True,
                    'resolver': trimesh.resolvers.FilePathResolver(texture_dir)
            }

            
            # Save the mesh in the requested format
            mesh.export(output_path, file_type=format, **export_options)
            
            print(f"Model exported successfully to {output_path}")
            
            # Save a copy as a temporary file for Gradio display
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{format}")
            tmp_path = tmp_file.name
            tmp_file.close()  # Close file to avoid permission issues
            
            print(f"Copying to temporary file {tmp_path}")
            shutil.copy2(output_path, tmp_path)
            
            self.last_model_path = output_path
            processing_time = time.time() - start
            print(f"Enhanced 3D model generation completed in {processing_time:.2f} seconds")
            
            return tmp_path, output_path, processing_time
        except Exception as e:
            print(f"Error generating 3D model: {e}")
            traceback.print_exc()
            return None, None, 0

    def clean_temp_files(self):
        """Clean up temporary files"""
        temp_dir = tempfile.gettempdir()
        for file in os.listdir(temp_dir):
            if file.endswith((".glb", ".obj", ".stl")):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass

# Create an enhanced Gradio interface
def create_enhanced_interface():
    # Initialize the enhanced model generator
    generator = EnhancedModelGenerator()
    
    # Define custom CSS for background image
    custom_css = """
    body {
        background-image: url('File=gradiobg.jpeg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Make containers slightly transparent to see background */
    .gradio-container {
        background-color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Optional: Add some styling for better readability */
    .gradio-container .prose {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 15px;
        border-radius: 10px;
    }
    """
    
    # Define theme
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    )
    
    # Create the interface
    with gr.Blocks(theme=theme) as demo:
        gr.Markdown("""
        # Enhanced 3D Model Generator
        
        This application converts 2D images into realistic interactive 3D models using advanced image processing techniques.
        Upload an image of a machine, landscape, or object to generate a detailed 3D representation with realistic lighting and textures.
        
        ## Features:
        - Edge preservation for better detail
        - Normal map generation for improved lighting
        - Adaptive mesh refinement in detailed areas
        - Ambient occlusion for realistic shadows
        - Multi-level detail control
        """)
        
        with gr.Tabs():
            with gr.TabItem("Generate Model"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(
                            type="pil", 
                            label="Upload Image", 
                            sources=["upload", "clipboard"],
                            image_mode="RGB"
                        )
                        
                        with gr.Group():
                            gr.Markdown("### Model Parameters")
                            with gr.Accordion("Basic Settings", open=True):
                                resolution_slider = gr.Slider(
                                    minimum=50, 
                                    maximum=300, 
                                    value=150, 
                                    step=10, 
                                    label="Resolution (higher = more detailed but slower)"
                                )
                                height_scale = gr.Slider(
                                    minimum=0.1, 
                                    maximum=1.5, 
                                    value=0.5, 
                                    step=0.05, 
                                    label="Height Scale"
                                )
                                smoothing = gr.Slider(
                                    minimum=0, 
                                    maximum=3.0, 
                                    value=1.0, 
                                    step=0.1, 
                                    label="Smoothing"
                                )
                                contrast = gr.Slider(
                                    minimum=0.5, 
                                    maximum=2.0, 
                                    value=1.2, 
                                    step=0.1, 
                                    label="Contrast Enhancement"
                                )
                            
                            with gr.Accordion("Advanced Settings", open=False):
                                detail_level = gr.Slider(
                                    minimum=0.0, 
                                    maximum=2.0, 
                                    value=1.0, 
                                    step=0.1, 
                                    label="Detail Preservation Level"
                                )
                                edge_weight = gr.Slider(
                                    minimum=0.0, 
                                    maximum=1.0, 
                                    value=0.3, 
                                    step=0.05, 
                                    label="Edge Enhancement"
                                )
                                adaptive_mesh = gr.Checkbox(
                                    label="Adaptive Mesh Refinement", 
                                    value=True
                                )
                                calculate_ao = gr.Checkbox(
                                    label="Calculate Ambient Occlusion (slower)", 
                                    value=True
                                )
                                format_dropdown = gr.Dropdown(
                                    choices=["glb", "obj", "stl"], 
                                    value="glb", 
                                    label="Output Format"
                                )
                        
                        with gr.Row():
                            generate_btn = gr.Button("Generate Enhanced 3D Model", variant="primary", scale=2)
                            preview_toggle = gr.Checkbox(label="Auto-Preview Height Map", value=False, scale=1)
                        
                        with gr.Group():
                            status_text = gr.Textbox(label="Status", value="Ready to generate...", interactive=False)
                            processing_time = gr.Textbox(label="Processing Time", visible=True, interactive=False)
                            file_path = gr.Textbox(label="Download Path", visible=True, interactive=False)
                            
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("3D Model"):
                                output_3d = LitModel3D(
                                    label="Generated 3D Model", 
                                    visible=True,
                                    height=600
                                )
                            with gr.TabItem("Height Map Preview"):
                                height_map_preview = gr.Image(
                                    type="numpy",
                                    label="Height Map Preview",
                                    visible=True,
                                    height=400
                                )
            
            with gr.TabItem("Help & Examples"):
                gr.Markdown("""
                ## Tips for Best Results
                
                - **Image Quality**: Higher resolution images produce better models, especially with fine details.
                - **Lighting**: Images with well-defined shadows and highlights produce better 3D results.
                - **Subject Matter**: Objects with defined shapes and clear edges work best.
                - **Edge Enhancement**: Increase this parameter for mechanical parts and architectural elements.
                - **Detail Preservation**: Increase for fine textures, decrease for smoother results.
                
                ## Enhanced Parameters Explained
                
                ### Basic Settings
                - **Resolution**: Controls the detail level of the 3D model (50-300).
                - **Height Scale**: Controls how pronounced the height variations are (0.1-1.5).
                - **Smoothing**: Reduces noise but can blur fine details (0.0-3.0).
                - **Contrast**: Enhances differences between light and dark areas (0.5-2.0).
                
                ### Advanced Settings
                - **Detail Preservation**: Preserves high-frequency details while still smoothing noise (0.0-2.0).
                - **Edge Enhancement**: Uses edge detection to preserve sharp features (0.0-1.0).
                - **Adaptive Mesh**: Adds more triangles in detailed areas for better precision.
                - **Ambient Occlusion**: Calculates realistic self-shadowing for improved depth perception.
                
                ## Output Formats
                - **GLB**: Best for web viewing and most 3D software
                - **OBJ**: Compatible with most 3D editing software
                - **STL**: Ideal for 3D printing
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Example Images")
                        example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
                        os.makedirs(example_dir, exist_ok=True)
                        example_images = gr.Examples(
                            examples=[
                                os.path.join(example_dir, "example1.jpg") if os.path.exists(os.path.join(example_dir, "example1.jpg")) else None,
                                os.path.join(example_dir, "example2.jpg") if os.path.exists(os.path.join(example_dir, "example2.jpg")) else None
                            ],
                            inputs=input_img,
                            examples_per_page=4
                        )
        
        # Preview height map
        def preview_height_map(image, detail_level, edge_weight, smoothing, contrast):
            if image is None:
                return None, "Please upload an image first."
            
            try:
                # Generate height map
                height_map, normal_map = generator.create_heightmap_from_image(
                    image,
                    detail_level=detail_level,
                    contrast=contrast,
                    edge_weight=edge_weight,
                    normal_smoothing=smoothing
                )
                
                # Convert to visible format
                height_map_viz = (height_map * 255).astype(np.uint8)
                height_map_color = cv2.applyColorMap(height_map_viz, cv2.COLORMAP_VIRIDIS)
                
                # Convert from BGR to RGB for display
                height_map_color = cv2.cvtColor(height_map_color, cv2.COLOR_BGR2RGB)
                
                return height_map_color, "Height map preview generated."
            except Exception as e:
                traceback.print_exc()
                return None, f"Error generating preview: {str(e)}"
        
        # Auto-preview when parameters change
        def should_preview(image, detail_level, edge_weight, smoothing, contrast, preview_enabled):
            if preview_enabled and image is not None:
                return preview_height_map(image, detail_level, edge_weight, smoothing, contrast)
            return None, ""
            
        # Parameter change listeners
        for param in [detail_level, edge_weight, smoothing, contrast]:
            param.change(
                should_preview,
                inputs=[input_img, detail_level, edge_weight, smoothing, contrast, preview_toggle],
                outputs=[height_map_preview, status_text]
            )
        
        # Handle image upload
        input_img.upload(
            should_preview,
            inputs=[input_img, detail_level, edge_weight, smoothing, contrast, preview_toggle],
            outputs=[height_map_preview, status_text]
        )
        
        # Handle model generation
        def process_and_update_status(image, resolution, height_scale, detail_level, edge_weight, smoothing, 
                                     contrast, adaptive_mesh, calculate_ao, format):
            if image is None:
                return None, None, "Please upload an image first.", "0 seconds", ""
            
            try:
                print("\n=== Starting model generation process ===")
                print(f"Image type: {type(image)}")
                
                # Generate the enhanced model
                model_path, saved_path, proc_time = generator.run_model(
                    image, 
                    resolution=resolution,
                    height_scale=height_scale,
                    detail_level=detail_level,
                    edge_weight=edge_weight,
                    smoothing=smoothing,
                    contrast=contrast,
                    adaptive_mesh=adaptive_mesh,
                    calculate_ao=calculate_ao,
                    format=format
                )
                
                # Generate height map preview
                height_map, _ = generator.create_heightmap_from_image(
                    image,
                    detail_level=detail_level,
                    contrast=contrast,
                    edge_weight=edge_weight,
                    normal_smoothing=smoothing
                )
                
                # Convert to visible format
                # Convert to visible format
                height_map_viz = (height_map * 255).astype(np.uint8)
                height_map_color = cv2.applyColorMap(height_map_viz, cv2.COLORMAP_VIRIDIS)
                
                # Convert from BGR to RGB for display
                height_map_color = cv2.cvtColor(height_map_color, cv2.COLOR_BGR2RGB)
                
                if model_path:
                    status = f"Model generated successfully! Format: {format.upper()}"
                    proc_time_text = f"{proc_time:.2f} seconds"
                    return model_path, height_map_color, status, proc_time_text, saved_path
                else:
                    return None, height_map_color, "Error generating model.", "0 seconds", ""
            except Exception as e:
                traceback.print_exc()
                return None, None, f"Error: {str(e)}", "0 seconds", ""
        
        generate_btn.click(
            process_and_update_status,
            inputs=[
                input_img, resolution_slider, height_scale, detail_level, 
                edge_weight, smoothing, contrast, adaptive_mesh, 
                calculate_ao, format_dropdown
            ],
            outputs=[output_3d, height_map_preview, status_text, processing_time, file_path]
        )
        
        # Handle preview toggle
        preview_toggle.change(
            should_preview,
            inputs=[input_img, detail_level, edge_weight, smoothing, contrast, preview_toggle],
            outputs=[height_map_preview, status_text]
        )
        
        # Clean up function
        demo.load(generator.clean_temp_files)
        
    return demo

# Launch the application
if __name__ == "__main__":
    demo = create_enhanced_interface()
    print("Launching Gradio app...")
    
    # Launch with specific server parameters
    demo.launch(
        server_name="127.0.0.1",  # IMPORTANT: Use 127.0.0.1, not localhost
        server_port=7868,         # Change this port for each app
        share=False,
        debug=True,
        show_error=True
    )
    
    print("Gradio app should now be running")