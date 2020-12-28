import base64
import urllib
from uuid import uuid1

import PIL.Image
import torchvision.transforms as T
import matplotlib

# use Agg as backend to not show image in server：
# https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib
matplotlib.use('Agg')
from fastai.vision import *
from fastai.vision import load_learner
from flask import Flask, request, render_template
from core import FeatureLoss

learn = None


# singleton start
def load_pkl(self) -> Any:
    global learn
    path = Path(".")
    learn = load_learner(path, 'ArtLine_650.pkl')


PklLoader = type('PklLoader', (), {"load_pkl": load_pkl})
pl = PklLoader()
pl.load_pkl()


# singleton end

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
        img_t = T.ToTensor()(img)
        img_fast = Image(img_t)

        p, img_hr, b = learn.predict(img_fast)
        r = Image(img_hr)
        demo_show(r, figsize=(8, 8), out_file=local_filename)
        result_img_base64 = read_img_file_as_base64('result/' + local_filename)
    except Exception  as e:
        return render_template('result.html', error=True)
    finally:
        if os.path.exists(local_file):
            os.remove(local_file)

    return render_template('result.html', error=False, result_img=result_img_base64)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
