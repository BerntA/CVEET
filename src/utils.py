import matplotlib
import matplotlib.pyplot as plt

def show_image(img, grayscale=False, file=None, size=(12,8)):
    plt.figure(figsize=size)
    ax = plt.subplot(111)
    if grayscale:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)        
    else:
        plt.imshow(img)
    plt.xticks([]),plt.yticks([])
    ax.axis("off")
    plt.tight_layout()
    
    if file:
        plt.savefig('../images/temp/{}.png'.format(file))
    
    plt.show()

def parseBBOXFile(file):
    with open(file, 'r') as f:
        content = []
        for i, line in enumerate(f):
            if i <= 12: # Skip these lines.
                continue
            line = line.strip().lower().replace('\n', '').replace('\r', '').replace('\t', '')
            line = line.replace('</object>', '')
            content.append(line)
        return ''.join(content[:-1]).split('<object>')[1:] # Return content - last line.
    
def getTagValue(text, tag):
    start, end = text.find('<{}>'.format(tag)), text.find('</{}>'.format(tag))
    return text[(start+len(tag)+2):end]
