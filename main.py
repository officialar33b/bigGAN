import torch 
# from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample, save_as_images, display_in_terminal
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample
from save_and_vis import convert_output_to_images, save_as_images, display_images

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BigGAN.from_pretrained('biggan-deep-512').to(device)

# making an input 
classes = ['ant', 'boxer', 'ambulance', 'tiger']
trunc = 0.4
class_vector = one_hot_from_names(classes, batch_size=4)
noise_vector = truncated_noise_sample(truncation=trunc, batch_size=4)

class_vector, noise_vector = torch.from_numpy(class_vector).to(device), torch.from_numpy(noise_vector).to(device)

with torch.no_grad():
    output = model(noise_vector, class_vector, trunc)

output = output.to('cpu')
# display_in_terminal(output)
# save_as_images(output)


# The regular convert function doesn't work.
# because you can't import 'PIL.Image'.

save_as_images(output)
display_images(output, classes)



