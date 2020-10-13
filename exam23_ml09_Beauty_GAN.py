#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


# In[2]:


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')


# In[4]:


# 이미지 로드
img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.show()


# In[5]:


img_result = img.copy()
dets = detector(img)             # detector: 주어진 이미지에서 얼굴을 찾아줌
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1,figsize=(16,10))
    for det in dets:               # dets에는 이미지에 존재하는 얼굴 개수만큼 값(fig,ax,넓이,높이)들이 저장되어있음
        x,y,w,h = det.left(), det.top(), det.width(), det.height()
        rect = patches.Rectangle((x,y),w,h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.imshow(img_result)
    plt.show()                     # 인지한 얼굴들을 빨간 사각형으로 표시


# In[6]:


fig, ax = plt.subplots(1, figsize=(16,10))
objs = dlib.full_object_detections()         # 이미지를 input하면, 그 이미지에 존재하는 모든 위치정보를 가져와서 저장      
for detection in dets:
    s = sp(img, detection)       # 이미지에 있는 하나의 얼굴의 landmark(총 5개)를 s에 저장
    objs.append(s)               # 각 얼굴들에 대한 landmark 정보가 objs에 모두 저장되게함
    for point in s.parts():
        circle = patches.Circle((point.x, point.y), radius=3, edgecolor='r', facecolor='r')
        ax.add_patch(circle)
ax.imshow(img_result)


# In[7]:


faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)         # 얼굴만 따로 뽑아줌   # padding: 얼굴 주위 여유분
fig, axes = plt.subplots(1, len(faces)+1, figsize=(20,16))
axes[0].imshow(img)
for i, face in enumerate(faces):
    axes[i+1].imshow(face)


# In[12]:


# 함수 선언
def align_faces(img):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)
    return faces

# 함수 호출
test_img = dlib.load_rgb_image('./imgs/01.jpg')
test_faces = align_faces(test_img)
fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))
axes[0].imshow(test_img)
for i, face in enumerate(test_faces):
    axes[i+1].imshow(face)


# In[20]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


# In[28]:


def preprocess(img):
    return (img/255 -0.5)*2
def deprocess(img):
    return (img+1)/2


# In[49]:


# source 이미지
img1 = dlib.load_rgb_image('./imgs/12.jpg')
img1_faces = align_faces(img1)
# reference 이미지
img2 = dlib.load_rgb_image('./imgs/makeup/002.jpg')
img2_faces = align_faces(img2)

fig, axes = plt.subplots(1,2,figsize=(16,10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
plt.show()


# In[50]:


src_img = img1_faces[0]
ref_img = img2_faces[0]

X_img = preprocess(src_img)
X_img = np.expand_dims(X_img, axis=0)

Y_img = preprocess(ref_img)
Y_img = np.expand_dims(Y_img, axis=0)

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})
output_img = deprocess(output[0])

fig, axes = plt.subplots(1,3,figsize=(20,10))
axes[0].set_title('Source')
axes[0].imshow(src_img)
axes[1].set_title('Reference')
axes[1].imshow(ref_img)
axes[2].set_title('Result')
axes[2].imshow(output_img)
plt.show()


# In[ ]:




