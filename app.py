import datetime
import os
from uuid import uuid1
import base64

from flask import Flask, request, render_template
import urllib

import PIL.Image
import torchvision.transforms as T
from fastai.vision import *
from fastai.vision import load_learner


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w ** 2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


def demo_show(the_img: Image, ax: plt.Axes = None, figsize: tuple = (3, 3), title: Optional[str] = None,
              hide_axis: bool = True,
              cmap: str = None, y: Any = None, out_file: str = None, **kwargs):
    "Show image on `ax` with `title`, using `cmap` if single-channel, overlaid with optional `y`"
    cmap = ifnone(cmap, defaults.cmap)
    ax = show_image(the_img, ax=ax, hide_axis=hide_axis, cmap=cmap, figsize=figsize)
    if y is not None: y.show(ax=ax, **kwargs)
    if title is not None: ax.set_title(title)
    ax.get_figure().savefig('result/' + out_file)
    plt.close(ax.get_figure())
    print('close')


if not (os.path.exists('./ArtLine_650.pkl')):
    MODEL_URL = "https://www.dropbox.com/s/starqc9qd2e1lg1/ArtLine_650.pkl?dl=1"
    urllib.request.urlretrieve(MODEL_URL, "ArtLine_650.pkl")

path = Path(".")
learn = load_learner(path, 'ArtLine_650.pkl')

app = Flask(__name__)


@app.route('/index', methods=['GET'])
@app.route('/', methods=['GET'])
def index_view():
    return render_template('index.html')


def read_img_file_as_base64(local_file) -> str:
    with open(local_file, "rb") as rf:
        base64_str = base64.b64encode(rf.read())
        os.remove(local_file)
        return base64_str.decode()




@app.route('/result', methods=["POST"])
def result_view():
    f = request.files['uimg']
    if f is None:
        return render_template('result.html', error=True)

    local_filename = '{0}{1}'.format(uuid1().hex, f.filename[f.filename.index('.'):len(f.filename)])
    local_file = 'tmp/{0}{1}'.format(uuid1().hex, f.filename[f.filename.index('.'):len(f.filename)])
    f.save(local_file)

    try:
        img = PIL.Image.open(local_file)
        # img = PIL.Image.open(BytesIO(f)).convert("RGB")
        img_t = T.ToTensor()(img)
        img_fast = Image(img_t)

        p, img_hr, b = learn.predict(img_fast)
        r = Image(img_hr)
        # r.show(figsize=(8,8))
        demo_show(r, figsize=(8, 8), out_file=local_filename)
        result_img_base64 = read_img_file_as_base64('result/' + local_filename)
        print('done')
    except Exception  as e:
        return render_template('result.html', error=True)
    finally:
        if os.path.exists(local_file):
            os.remove(local_file)

    return render_template('result.html', error=False, result_img=result_img_base64)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
