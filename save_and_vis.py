from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt  
import os 

def convert_output_to_images(obj):
    """ Convert an output tensor from BigGAN in a list of images.
        Params:
            obj: tensor or numpy array of shape (batch_size, channels, height, width)
        Output:
            list of Pillow Images of size (height, width)
    """

    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()

    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(Image.fromarray(out_array))
    return img

def save_as_images(obj, file_name='./results/output'):
    """ Convert and save an output tensor from BigGAN in a list of saved images.
        Params:
            obj: tensor or numpy array of shape (batch_size, channels, height, width)
            file_name: path and beggingin of filename to save.
                Images will be saved as `file_name_{image_number}.png`
    """
    img = convert_output_to_images(obj)

    for i, out in enumerate(img):
        # current_file_name = file_name + '_%d.png' % (i+1)
        current_file_name = file_name + '{}.png'.format(i+1)
        if os.path.exists('./results'):
            pass 
        else: 
            os.mkdir('./results')

        out.save(current_file_name, 'png')

def display_images(obj, classes):
    img = convert_output_to_images(obj)
    img = [np.array(image)/255.0 for image in img]
    fig = plt.figure(figsize=(50,50))
    rows, columns = 2,2
    for i, out in enumerate(img):
        fig.add_subplot(rows, columns, i+1)
        plt.title(classes[i])
        plt.axis('off')
        plt.imshow(out)


    plt.show()
