import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
# from ipywidgets import widgets
from scipy.sparse.linalg import svds
import streamlit as st
from PIL import Image
import cv2
from io import BytesIO

def svd(image_pil, k):
  image = np.array(image_pil)
  image  = image.astype(np.float64)
  R,G,B = image[:,:,0], image[:,:,1], image[:,:,2]

  def compress(A,k):
    U,S,Vt = np.linalg.svd(A,full_matrices=False)
    # sigma = np.diag(S)

    Uk= U[:,:k]
    sigmak= np.diag(S[:k])
    # Vtk = Vt[:k] gotta mention columns also
    Vtk = Vt[:k,:]

    recon_img = Uk@sigmak@Vtk
    return recon_img #but for diff channels
  R_compress = compress(R,k)
  G_compress= compress(G,k)
  B_compress = compress(B,k)

  com_img = np.stack((R_compress,G_compress,B_compress), axis=2).astype(np.uint8)
  com_img = np.clip(com_img,0,255) #really important
  return com_img

#Streamlit starts here
st.title("Compress your image using SVD")
st.image("https://github.com/kartavayv/svd/blob/main/image_comparison.png","", use_container_width=True)
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpeg','jpg'])

if uploaded_file:
  image = Image.open(uploaded_file)
  st.image(image, caption="Your image", use_container_width=True)

  k = st.slider("Select the compression level/adjust image quality", 1, max_value= 500,step=5, value=50)

  if st.button("Compress Image"):
    compressed_image = svd(image,k)
    st.image(compressed_image, caption=f"Compressed image below (k={k})", use_container_width=True)
    st.write("DOWNLOAD THE IMAGE BELOW")
    _, buffer = cv2.imencode(".jpg", compressed_image)
    compressed_bytes = BytesIO(buffer.tobytes())
    st.download_button(
      label="📥Download your image here, man!",
      data = compressed_bytes,
      file_name = "compressed.jpg",
      mime = "image/jpeg"
    )
