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


if not (os.path.exists('./ArtLine_650.pkl')):
    MODEL_URL = "https://www.dropbox.com/s/starqc9qd2e1lg1/ArtLine_650.pkl?dl=1"
    urllib.request.urlretrieve(MODEL_URL, "ArtLine_650.pkl")
path = Path(".")
learn = load_learner(path, 'ArtLine_650.pkl')

url = 'https://rmt.dogedoge.com/fetch/~/source/unsplash/photo-1568602471122-7832951cc4c5?ixid=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=1650&q=80'  # @param {type:"string"}

response = requests.get(url)
img = PIL.Image.open(BytesIO(response.content)).convert("RGB")
img_t = T.ToTensor()(img)
img_fast = Image(img_t)


def demo_show(the_img: Image, ax: plt.Axes = None, figsize: tuple = (3, 3), title: Optional[str] = None,
              hide_axis: bool = True,
              cmap: str = None, y: Any = None, **kwargs):
    "Show image on `ax` with `title`, using `cmap` if single-channel, overlaid with optional `y`"
    cmap = ifnone(cmap, defaults.cmap)
    ax = show_image(the_img, ax=ax, hide_axis=hide_axis, cmap=cmap, figsize=figsize)
    if y is not None: y.show(ax=ax, **kwargs)
    if title is not None: ax.set_title(title)
    ax.get_figure().savefig("./output.png")


p, img_hr, b = learn.predict(img_fast)
r = Image(img_hr)
# r.show(figsize=(8,8))
demo_show(r, figsize=(8, 8))
print('done')
