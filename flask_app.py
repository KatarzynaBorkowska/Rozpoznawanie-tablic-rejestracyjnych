from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # Wyrenderuj szablon z formularzem
    return render_template('index.html')

@app.route('/', methods=['POST'])
def index_post():
    # Pobierz dane przesłane z formularza
    form_data = request.form['samochod']
    print(form_data)
    # Wyrenderuj szablon z przyciskiem powrotu
    return render_template('return.html')

@app.route('/')
def return_page():
    # Wyrenderuj stronę główną
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
