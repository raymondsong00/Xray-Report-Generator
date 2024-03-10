import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.text as txt
import pandas as pd
import numpy as np
from PIL import Image

fig, ax = plt.subplots()
# real_rep = pd.read_json('/home/llm-hackathon/LLaVA/muchan/', lines=True)
gen_rep = pd.read_json('/home/llm-hackathon/LLaVA/muchan/testset_answer_finding_impression.jsonl', lines=True)

image_fp = '/data/UCSD_cxr/jpg/'

def norm_img(image_data):
    return ((image_data - image_data.min()) / (image_data.ptp()) * 255).astype(np.uint8)

i = 0
report_text = ax.text(200, 200, s=gen_rep.loc[i, 'text'])
plt.ioff()
def display_img():
    global i
    ax.clear()
    im_fp = '/data/UCSD_cxr/jpg/' + gen_rep.loc[i, 'question_id']
    img = Image.open(im_fp)
    n_img = norm_img(np.asarray(img))
    report_text.set(text=gen_rep.loc[i, 'text'])
    plt.imshow(n_img)
    ax.axis('off')
    plt.show()
    i += 1


display_img()
next_button = widgets.Button(ax, label='Next')
next_button.on_clicked(display_img)
# ax.plot(range(0,4), range(10, 14))
plt.show()
