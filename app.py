from flask import Flask, render_template, redirect, request
from generate import generate
from models import Decoder, Encoder
from configs import get_args
app = Flask(__name__)
# encoder = Encoder(4, 64).to(device)
# decoder = Decoder(4, 64).to(device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emoji/', methods=['GET', 'POST'])
def emoji():

    # if request.method != 'POST':
    #     return redirect('/')
    # emoji_pred = generate()
    # эмодзи нужно сохранить в папку static
    filename = 'nikolay.jpg'
    return render_template('emoji.html', filename=filename)

app.run(debug=True)