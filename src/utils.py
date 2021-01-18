import matplotlib
import matplotlib.pyplot as plt

def show_image(img, grayscale=False, file=None):
    plt.figure(figsize=(12,8))
    ax = plt.subplot(111)
    if grayscale:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)        
    else:
        plt.imshow(img)
    plt.xticks([]),plt.yticks([])
    ax.axis("off")
    plt.tight_layout()
    
    if file:
        plt.savefig('../data/temp/{}.png'.format(file))
    
    plt.show()