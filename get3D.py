#----------------------------------------------------#
#   利用deeplabV3+模型对图像序列进行预测并三维重建
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#----------------------------------------------------#
import time
import os
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from deeplab import DeeplabV3
import pydicom
from skimage import measure
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mayavi import mlab
from stl import mesh


if __name__ == "__main__":
    deeplab = DeeplabV3()
    # dir_origin_path 指向包含所有子文件夹的路径
    dir_origin_path = "save_pic"
    # 对每个子文件夹进行处理
    for name in os.listdir(dir_origin_path):
        if os.path.isdir(os.path.join(dir_origin_path, name)) and name == "03.02 l2-r":  # 检查是否为文件夹
            print("处理子文件夹: ", name)
            
            sub_dir_origin_path = dir_origin_path + "/" + name + "/"
            dir_save_path = "img_out" + "/" + name + "/"

        
            img_names = os.listdir(sub_dir_origin_path)
            # 记录当前时间
            start = time.time()
            for img_name in tqdm(img_names):
                if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path  = os.path.join(sub_dir_origin_path, img_name)
                    image       = Image.open(image_path)
                    r_image     = deeplab.detect_image(image)
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    r_image.save(os.path.join(dir_save_path, img_name))
            imgfile = []
            #opencv读取所有预测结果图片
            for i in range(len(img_names)):
                img = cv2.imread(dir_save_path + img_names[i],0)
                imgfile.append(img)
            
            # 创建3D numpy数组
            img_shape = list(imgfile[0].shape)

            img_shape.append(len(imgfile))
            img3d = np.zeros(img_shape)

            # 将所有图片的像素值存入3D数组
            for i in range(len(imgfile)):
                # resize为img_shape
                imgfile[i] = cv2.resize(imgfile[i], (img_shape[1], img_shape[0]))
                img3d[:, :, i] = imgfile[i]
            
            #计算img3d中非零体素的个数
            print(np.count_nonzero(img3d))

            #一个pixel的体素大小
            pixel_area = (2.383/240) * (3.525/355) * 0.01 
            #计算体积。单位mm^3
            volume = np.count_nonzero(img3d) * pixel_area
            print("volume:", volume)

            img3d = gaussian_filter(img3d, sigma=1.0)

            


            
            
            # 记录当前时间
            end = time.time()
            print("完整重建时间：", end - start)
            # 绘制等值面
            verts, faces, _, _ = measure.marching_cubes(img3d, 0 ,spacing=(0.1, 0.1, 0.1))
            verts = verts.T

            if volume > 0.3:
                mlab.show()
            
            else:
                continue

            

            print(verts.shape)
            print(faces.shape)
            mlab.triangular_mesh([verts[0]], [verts[1]], [verts[2]], faces)
            # 显示平滑化
            mlab.pipeline.surface(mlab.pipeline.triangular_mesh_source([verts[0]], [verts[1]], [verts[2]], faces))
           
            mlab.show()

            start = time.time()

            # 保存为stl文件
            # 创建一个新的空的三角网格
            your_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces):
                for j in range(3):
                    your_mesh.vectors[i][j] = verts[:, f[j]]
            # 保存为stl文件，名称为name
            your_mesh.save('stl_result/' + name + '.stl')
            # 计算重建时间
            end = time.time()
            print("保存模型时间：", end - start)

            # 保存name和对应的体积信息
            with open('volume.txt', 'a') as f:
                f.write(name + ' ' + str(volume) + '\n')

