
global content_filename,style_filename;



def artisticImage():
    tkinter.messagebox.showinfo("Process","Processing Image please wait")
    print(content_filename)
    print(style_filename)
    content_path = tf.keras.utils.get_file('belfry.jpg','https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/belfry-2611573_1280.jpg')
    style_path = tf.keras.utils.get_file('style23.jpg','https://storage.googleapis.com/khanhlvg-public.appspot.com/arbitrary-style-transfer/style23.jpg')

    style_predict_path = tf.keras.utils.get_file('style_predict.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite')
    style_transform_path = tf.keras.utils.get_file('style_transform.tflite', 'https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite')

    def load_img(path_to_img):
      img = tf.io.read_file(path_to_img)
      img = tf.io.decode_image(img, channels=3)
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = img[tf.newaxis, :]

      return img

    def preprocess_image(image, target_dim):
      shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
      short_dim = min(shape)
      scale = target_dim / short_dim
      new_shape = tf.cast(shape * scale, tf.int32)
      image = tf.image.resize(image, new_shape)

      image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)

      return image
                                                   
                                                   
    content_image = load_img(content_filename)
    style_image = load_img(style_filename)

    # Preprocess the input images.
    preprocessed_content_image = preprocess_image(content_image, 384)
    preprocessed_style_image = preprocess_image(style_image, 256)

    def run_style_predict(preprocessed_style_image):
      # Load the model.
      interpreter = tf.lite.Interpreter(model_path=style_predict_path)

      # Set model input.
      interpreter.allocate_tensors()
      input_details = interpreter.get_input_details()
      interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

      # Calculate style bottleneck.
      interpreter.invoke()
      style_bottleneck = interpreter.tensor(
          interpreter.get_output_details()[0]["index"]
          )()

      return style_bottleneck

    # Calculate style bottleneck for the preprocessed style image.
    style_bottleneck = run_style_predict(preprocessed_style_image)


    def run_style_transform(style_bottleneck, preprocessed_content_image):
      # Load the model.
      interpreter = tf.lite.Interpreter(model_path=style_transform_path)

      # Set model input.
      input_details = interpreter.get_input_details()
      interpreter.allocate_tensors()

      # Set model inputs.
      interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
      interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
      interpreter.invoke()

      # Transform content image.
      stylized_image = interpreter.tensor(
          interpreter.get_output_details()[0]["index"]
          )()

      return stylized_image

    # Stylize the content image using the style bottleneck.
    stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image)

    # Visualize the output.
    stylized_image = (stylized_image.squeeze()*255).astype(np.uint8)
    img = Image.fromarray(stylized_image,'RGB')
    img.save('stylized_image.png')
    img.show()
    

    style_bottleneck_content = run_style_predict(
        preprocess_image(content_image, 256)
        )
    content_blending_ratio = 0.5 

    # Blend the style bottleneck of style image and content image
    style_bottleneck_blended = content_blending_ratio * style_bottleneck_content + (1 - content_blending_ratio) * style_bottleneck;

    # Stylize the content image using the style bottleneck.
    stylized_image_blended = run_style_transform(style_bottleneck_blended,
                                                 preprocessed_content_image)

    # Visualize the output.
    stylized_image_blended = (stylized_image_blended.squeeze()*255).astype(np.uint8)
    img = Image.fromarray(stylized_image_blended,'RGB')
    img.save('stylized_image_blended.png')
    img.show()


def selectContentDirectory():
    global content_filename
    content_filename = askopenfilename()
def selectStyleDirectory():
    global style_filename
    style_filename = askopenfilename()
   

def addButtons():
    button1 = tkinter.Button(root,text="Content image",command=selectContentDirectory)
    button1.place(relx=0.33,rely=0.7,anchor=CENTER)

    button2 = tkinter.Button(root,text="Style Image",command=selectStyleDirectory)
    button2.place(relx=0.66,rely=0.7,anchor=CENTER)

    button3 = tkinter.Button(root,text="Create",command=artisticImage)
    button3.place(relx=0.5,rely=0.8,anchor=CENTER)


def addImages(root):
    startframe = tkinter.Frame(root)
    
    canvas = tkinter.Canvas(startframe,width=250,height=250)
    startframe.place(relx = 0.5, rely = 0.3, anchor = CENTER)
    
    canvas.pack()
    root.one=one=tkinter.PhotoImage(file=r'Logo.png')
    canvas.create_image((0,0), image=one, anchor=NW)
    
from tkinter import *
import tkinter 
import tkinter.messagebox as tkMessageBox
from tkinter.filedialog import askopenfilename
import tensorflow as tf
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
from PIL import Image

root = tkinter.Tk()
root.title("Image Blending tool")
root.geometry("500x500+200+100")
addButtons()
addImages(root)
root.mainloop()
