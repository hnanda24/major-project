import base64
import pandas as pd
from arrow import now
from glob import glob
from img2vec_pytorch import Img2Vec
from io import BytesIO
from os.path import basename
from PIL import Image
from plotly import express
from IPython.display import publish_display_data

# we're going to use the updated dataset
GLOB = 'Aero-engine_defect-detect_new/images'
SIZE = 512
STOP = 500
THUMBNAIL_SIZE = (100, 100)


def embed(model, filename: str):
    with Image.open(fp=filename, mode='r') as image:
        return model.get_vec(image, tensor=True).numpy().reshape(SIZE,)


# https://stackoverflow.com/a/952952
def flatten(arg):
    return [x for xs in arg for x in xs]

def png(filename: str) -> str:
    with Image.open(fp=filename, mode='r') as image:
        buffer = BytesIO()
        # our images are pretty big; let's shrink the hover images to thumbnail size
        image.resize(size=THUMBNAIL_SIZE).save(buffer, format='png')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

def get_picture_from_glob(arg: str, tag: str, stop: int) -> list:
    time_get = now()
    result = [pd.Series(data=[tag, basename(input_file), embed(model=model, filename=input_file), png(filename=input_file), ],
                        index=['tag', 'name', 'value', 'image'])
        for index, input_file in enumerate(glob(pathname=arg)) if index < stop]
    print('encoded {} data {} rows in {}'.format(tag, len(result), now() - time_get))
    return result

time_start = now()
model = Img2Vec(cuda=False, model='resnet-18')
data_dict = {basename(folder) : folder + '/*.jpg' for folder in glob(GLOB + '/*')}
df = pd.DataFrame(data=flatten(arg=[get_picture_from_glob(arg=value, tag=key, stop=STOP) for key, value in data_dict.items()]))
# we need to add a key so we can join the labels
df['label name'] = df['name'].apply(func=lambda x: x.replace('.jpg', '.txt'))
print('done in {}'.format(now() - time_start))

root = 'Aero-engine_defect-detect_new/labels/'
dfs = []
for subfolder in ['val', 'train']:
    for input_file in glob(pathname=root + subfolder + '/*.txt'):
        current_df = pd.read_csv(filepath_or_buffer=input_file, sep=' ', names=['label', 'w0', 'w1', 'w2', 'w3'])
        current_df['tag'] = [subfolder] * len(current_df)
        current_df['name'] = basename(input_file)
        dfs.append(current_df)
labels_df = pd.concat(objs=dfs, axis=0)
labels_df.head()

all_df = df.merge(right=labels_df, left_on='label name', right_on='name', how='left')

all_df['class'] = all_df['label'].map({0: 'scratch', 1: 'dot', 2: 'crease', 3: 'damage'})

all_df.head()

val_df = all_df[(all_df['tag_x'] == 'val') & (all_df['tag_y'] == 'val')]
train_df = all_df[(all_df['tag_x'] == 'train') & (all_df['tag_y'] == 'train')]
train_df.head()

express.histogram(data_frame=train_df, x='class')

from arrow import now
from umap import UMAP

time_start = now()
umap = UMAP(random_state=2024, verbose=True, n_jobs=1, low_memory=False, n_epochs=2000)
all_df[['x', 'y']] = umap.fit_transform(X=all_df['value'].apply(pd.Series))
print('done with UMAP in {}'.format(now() - time_start))

from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.palettes import Category20_4

from bokeh.plotting import figure
from bokeh.plotting import output_notebook
from bokeh.plotting import show
from bokeh.transform import factor_cmap

output_notebook()

plot_df = all_df.dropna(subset=['class'])[['x', 'y', 'class', 'image']]

datasource = ColumnDataSource(plot_df)
mapper = factor_cmap('class', palette=Category20_4, factors=['damage', 'crease', 'scratch', 'dot'])

plot_figure = figure(title='UMAP projection: engine defects', width=1000, height=800, tools=('pan, wheel_zoom, reset'))

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Class:</span>
        <span style='font-size: 18px'>@class</span>
    </div>
</div>
"""))

plot_figure.circle('x', 'y', source=datasource, color=mapper, line_alpha=0.6, fill_alpha=0.6, size=5,)
show(plot_figure)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest = RandomForestClassifier(random_state=2024, n_estimators=20)
forest.fit(X=train_df['value'].apply(pd.Series), y=train_df['class'])

print('accuracy: {:5.4f}'.format(accuracy_score(y_true=val_df['class'], y_pred=forest.predict(X=val_df['value'].apply(pd.Series)))))

from sklearn.metrics import classification_report
print(classification_report(y_true=val_df['class'], y_pred=forest.predict(X=val_df['value'].apply(pd.Series))))



import pickle

# Save the trained model to a file
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(forest, file)

# Now the model is saved as 'random_forest_model.pkl' in your current directory
