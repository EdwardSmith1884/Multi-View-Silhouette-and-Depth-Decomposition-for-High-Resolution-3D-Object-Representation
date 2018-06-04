import numpy as np
import os 
def call(command):
    os.system('%s > /dev/null 2>&1' % command)

# this method creates a mesh copy of a voxel model
# it is efficinet as only exposed faces are drawn
#`could be written a tad better, but I think this way is the easiest to read 
def voxel2mesh(voxels, threshold=0):
    # these faces and verticies define the various sides of a cube
    top_verts = [[0,0,1], [1,0,1], [1,1,1], [0,1,1]]    
    top_faces = [[1,2,4], [2,3,4]] 

    bottom_verts = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]    
    bottom_faces = [[2,1,4], [3,2,4]] 

    left_verts = [[0,0,0], [0,0,1], [0,1,0], [0,1,1]]
    left_faces = [[1,2,4], [3,1,4]]

    right_verts = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]
    right_faces = [[2,1,4], [1,3,4]]

    front_verts = [[0,1,0], [1,1,0], [0,1,1], [1,1,1]]
    front_faces = [[2,1,4], [1,3,4]]

    back_verts = [[0,0,0], [1,0,0], [0,0,1], [1,0,1]]
    back_faces = [[1,2,4], [3,1,4]]

    top_verts = np.array(top_verts)
    top_faces = np.array(top_faces)
    bottom_verts = np.array(bottom_verts)
    bottom_faces = np.array(bottom_faces)
    left_verts = np.array(left_verts)
    left_faces = np.array(left_faces)
    right_verts = np.array(right_verts)
    right_faces = np.array(right_faces)
    front_verts = np.array(front_verts)
    front_faces = np.array(front_faces)
    back_verts = np.array(back_verts)
    back_faces = np.array(back_faces)

    dim = voxels.shape[0]
    new_voxels = np.zeros((dim+2, dim+2, dim+2))
    new_voxels[1:dim+1,1:dim+1,1:dim+1 ] = voxels
    voxels= new_voxels


    scale = 0.01
    cube_dist_scale = 1.
    verts = []
    faces = []
    curr_vert = 0
    a,b,c= np.where(voxels>threshold)
    for i,j,k in zip(a,b,c):
        #top
        if voxels[i,j,k+1]<threshold: 
            verts.extend(scale * (top_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(top_faces + curr_vert)
            curr_vert += len(top_verts)
    
        #bottom
        if voxels[i,j,k-1]<threshold: 
            verts.extend(scale * (bottom_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(bottom_faces + curr_vert)
            curr_vert += len(bottom_verts)
            
        #left
        if voxels[i-1,j,k]<threshold: 
            verts.extend(scale * (left_verts+ cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(left_faces + curr_vert)
            curr_vert += len(left_verts)
            
        #right
        if voxels[i+1,j,k]<threshold: 
            verts.extend(scale * (right_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(right_faces + curr_vert)
            curr_vert += len(right_verts)
            
        #front
        if voxels[i,j+1,k]<threshold: 
            verts.extend(scale * (front_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(front_faces + curr_vert)
            curr_vert += len(front_verts)
            
        #back
        if voxels[i,j-1,k]<threshold: 
            verts.extend(scale * (back_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(back_faces + curr_vert)
            curr_vert += len(back_verts)
            
    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def voxel2obj(filename, pred, show='False', threshold=0.4):
    verts, faces = voxel2mesh(pred, threshold )
    
    write_obj(filename, verts, faces)
    call('meshlabserver -i ' + filename + ' -o ' + filename + ' -s scripts/mesh.mlx')
    call('meshlab ' + filename)
