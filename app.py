from flask import Flask ,render_template

app = Flask(__name__)

@app.route('/')
def signup():
    return render_template('signup.html')

@app.route('/museums')
def museums():
      return render_template('museums.html') 

@app.route('/contact')
def contact():
      return render_template('contact.html') 

@app.route('/about')
def about():
      return render_template('about.html')
 
@app.route('/index')
def home():
      return render_template('index.html') 

@app.route('/coin')
def coin():
      return render_template('coin.html') 

@app.route('/login')
def login():
      return render_template('login.html') 

@app.route('/signup')
def sig():
    return render_template('signup.html')


if __name__ =="__main__": 
    app.run(debug=True)