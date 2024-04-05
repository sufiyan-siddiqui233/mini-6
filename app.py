##  WORKING CODE

import pickle
from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd
import requests
from PIL import Image 

# Load the Random Forest model
rf_model = load('random_forest_model3.pkl')

# Initialize Flask application
app = Flask(__name__)

# Preprocessing function for categorical variables
def preprocess_inputs(autoPay, topRatedListing, bestOfferEnabled, buyItNowAvailable,
                      expeditedShipping, seller_in_us, gold, silver, Less_than_week,
                      Less_than_month, more_than_year, high_quality_words,
                      low_quality_words, Calculated, Free, FreePickup, Auction,
                      FixedPrice, bronze, copper, augustus, commodus, domitian,
                      macrinus, nero, nerva, otho, philip, septimius_severus,
                      tiberius, valerian, none_of_above, other_ship):
    

    
    return [autoPay, topRatedListing, bestOfferEnabled, buyItNowAvailable,
                      expeditedShipping, seller_in_us, gold, silver, Less_than_week,
                      Less_than_month, more_than_year, high_quality_words,
                      low_quality_words, Calculated, Free, FreePickup, Auction,
                      FixedPrice, bronze, copper, augustus, commodus, domitian,
                      macrinus, nero, nerva, otho, philip, septimius_severus,
                      tiberius, valerian, none_of_above, other_ship]




# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

# Class labels
class_labels = {0: 'cat', 1: 'dog', 2: 'bird'}  # Example mapping








# Define routes

@app.route('/')
def signup():
    return render_template('signup.html')

@app.route('/login')
def login():  
    return render_template('login.html')

@app.route('/index')
def index():
    api_url = 'https://api.harvardartmuseums.org/exhibition?apikey=23b1e72f-ffcd-45f5-8a60-5d46b9a962cc'
    response = requests.get(api_url)
    data = response.json()
    return render_template('index.html')

@app.route('/museums')
def museums():
        # Fetch data from the API
    api_url = 'https://api.harvardartmuseums.org/image?apikey=23b1e72f-ffcd-45f5-8a60-5d46b9a962cc'
    response = requests.get(api_url)
    data = response.json()  # Assuming the API returns JSON data
    
    # Render the fetched data on a web page
    return render_template('museums.html', data=data) 

@app.route('/coin')
def coins():
    return render_template('coin-prediction.html') 
@app.route('/contact')
def contact():
    return render_template('contact.html') 
@app.route('/about')
def about():
    return render_template('about.html') 


@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    auto_pay = request.form['autoPay']
    top_rated_listing = request.form['topRatedListing']
    best_offer_enabled = request.form['bestOfferEnabled']
    buy_it_now_available = request.form['buyItNowAvailable']
    expedited_shipping = request.form['expeditedShipping']
    seller_in_us = request.form['sellerInUS']
    gold = request.form['gold']
    silver = request.form['silver']
    Less_than_week = request.form['LessThanWeek']
    Less_than_month = request.form['LessThanMonth']
    more_than_year = request.form['MoreThanYear']
    high_quality_words = request.form['highQualityWords']
    low_quality_words = request.form['lowQualityWords']
    Calculated = request.form['Calculated']
    Free = request.form['Free']
    FreePickup = request.form['FreePickup']
    Auction = request.form['Auction']
    FixedPrice = request.form['FixedPrice']
    bronze = request.form['bronze']
    copper = request.form['copper']
    augustus = request.form['augustus']
    commodus = request.form['commodus']
    domitian = request.form['domitian']
    macrinus = request.form['macrinus']
    nero = request.form['nero']
    nerva = request.form['nerva']
    otho = request.form['otho']
    philip = request.form['philip']
    septimius_severus = request.form['septimius_severus']
    tiberius = request.form['tiberius']
    valerian = request.form['valerian']
    none_of_above = request.form['noneOfAbove']
    other_ship = request.form['otherShip']
    
    # Preprocess form inputs
    features = preprocess_inputs(auto_pay, top_rated_listing, best_offer_enabled, buy_it_now_available,
                                 expedited_shipping, seller_in_us, gold, silver, Less_than_week,
                                 Less_than_month, more_than_year, high_quality_words,
                                 low_quality_words, Calculated, Free, FreePickup, Auction,
                                 FixedPrice, bronze, copper, augustus, commodus, domitian,
                                 macrinus, nero, nerva, otho, philip, septimius_severus,
                                 tiberius, valerian, none_of_above, other_ship)

    # Make prediction using the model
    df = pd.DataFrame([features], columns=['autoPay', 'topRatedListing', 'bestOfferEnabled', 'buyItNowAvailable',
                      'expeditedShipping', 'seller_in_us', 'gold', 'silver', 'Less_than_week',
                      'Less_than_month', 'more_than_year', 'high_quality_words',
                      'low_quality_words', 'Calculated', 'Free', 'FreePickup', 'Auction',
                      'FixedPrice', 'bronze', 'copper', 'augustus', 'commodus', 'domitian',
                      'macrinus', 'nero', 'nerva', 'otho', 'philip', 'septimius severus',
                     'tiberius', 'valerian', 'none_of_above', 'other_ship'])

    prediction = rf_model.predict(df)
    
    return render_template('result.html', prediction=prediction[0])

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        image = Image.open(file)
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]
        return render_template('result2.html', label=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)


















# ##  WORKING CODE

# from flask import Flask, render_template, request
# from joblib import load
# import pandas as pd

# # Load the Random Forest model
# rf_model = load('random_forest_model3.pkl')

# # Initialize Flask application
# app = Flask(__name__)

# # Preprocessing function for categorical variables
# def preprocess_inputs(autoPay, topRatedListing, bestOfferEnabled, buyItNowAvailable,
#                       expeditedShipping, seller_in_us, gold, silver, Less_than_week,
#                       Less_than_month, more_than_year, high_quality_words,
#                       low_quality_words, Calculated, Free, FreePickup, Auction,
#                       FixedPrice, bronze, copper, augustus, commodus, domitian,
#                       macrinus, nero, nerva, otho, philip, septimius_severus,
#                       tiberius, valerian, none_of_above, other_ship):
    

    
#     return [autoPay, topRatedListing, bestOfferEnabled, buyItNowAvailable,
#                       expeditedShipping, seller_in_us, gold, silver, Less_than_week,
#                       Less_than_month, more_than_year, high_quality_words,
#                       low_quality_words, Calculated, Free, FreePickup, Auction,
#                       FixedPrice, bronze, copper, augustus, commodus, domitian,
#                       macrinus, nero, nerva, otho, philip, septimius_severus,
#                       tiberius, valerian, none_of_above, other_ship]


# # Define routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/museums')
# def museums():
#     return render_template('museums.html') 

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get form inputs
#     auto_pay = request.form['autoPay']
#     top_rated_listing = request.form['topRatedListing']
#     best_offer_enabled = request.form['bestOfferEnabled']
#     buy_it_now_available = request.form['buyItNowAvailable']
#     expedited_shipping = request.form['expeditedShipping']
#     seller_in_us = request.form['sellerInUS']
#     gold = request.form['gold']
#     silver = request.form['silver']
#     Less_than_week = request.form['LessThanWeek']
#     Less_than_month = request.form['LessThanMonth']
#     more_than_year = request.form['MoreThanYear']
#     high_quality_words = request.form['highQualityWords']
#     low_quality_words = request.form['lowQualityWords']
#     Calculated = request.form['Calculated']
#     Free = request.form['Free']
#     FreePickup = request.form['FreePickup']
#     Auction = request.form['Auction']
#     FixedPrice = request.form['FixedPrice']
#     bronze = request.form['bronze']
#     copper = request.form['copper']
#     augustus = request.form['augustus']
#     commodus = request.form['commodus']
#     domitian = request.form['domitian']
#     macrinus = request.form['macrinus']
#     nero = request.form['nero']
#     nerva = request.form['nerva']
#     otho = request.form['otho']
#     philip = request.form['philip']
#     septimius_severus = request.form['septimius_severus']
#     tiberius = request.form['tiberius']
#     valerian = request.form['valerian']
#     none_of_above = request.form['noneOfAbove']
#     other_ship = request.form['otherShip']
    
#     # Preprocess form inputs
#     features = preprocess_inputs(auto_pay, top_rated_listing, best_offer_enabled, buy_it_now_available,
#                                  expedited_shipping, seller_in_us, gold, silver, Less_than_week,
#                                  Less_than_month, more_than_year, high_quality_words,
#                                  low_quality_words, Calculated, Free, FreePickup, Auction,
#                                  FixedPrice, bronze, copper, augustus, commodus, domitian,
#                                  macrinus, nero, nerva, otho, philip, septimius_severus,
#                                  tiberius, valerian, none_of_above, other_ship)

#     # Make prediction using the model
#     df = pd.DataFrame([features], columns=['autoPay', 'topRatedListing', 'bestOfferEnabled', 'buyItNowAvailable',
#                       'expeditedShipping', 'seller_in_us', 'gold', 'silver', 'Less_than_week',
#                       'Less_than_month', 'more_than_year', 'high_quality_words',
#                       'low_quality_words', 'Calculated', 'Free', 'FreePickup', 'Auction',
#                       'FixedPrice', 'bronze', 'copper', 'augustus', 'commodus', 'domitian',
#                       'macrinus', 'nero', 'nerva', 'otho', 'philip', 'septimius severus',
#                      'tiberius', 'valerian', 'none_of_above', 'other_ship'])

#     prediction = rf_model.predict(df)
    
#     return render_template('result.html', prediction=prediction[0])

# if __name__ == '__main__':
#     app.run(debug=True)























# from flask import Flask, render_template, request
# from joblib import load
# import pandas as pd

# # Load the Random Forest model
# rf_model = load('random_forest_model3.pkl')

# # Initialize Flask application
# app = Flask(__name__)

# # Preprocessing function for categorical variables
# # Preprocessing function for categorical variables
# def preprocess_inputs(auto_pay, top_rated_listing, best_offer_enabled, buy_it_now_available,
#                       expedited_shipping, seller_in_us, metal_type, time, type_of_shipping,
#                       type_of_selling, emperor_name):
#     # Convert categorical variables to numerical format
#     auto_pay = 1 if auto_pay.lower() == 'yes' else 0
#     top_rated_listing = 1 if top_rated_listing.lower() == 'yes' else 0
#     best_offer_enabled = 1 if best_offer_enabled.lower() == 'yes' else 0
#     buy_it_now_available = 1 if buy_it_now_available.lower() == 'yes' else 0
#     expedited_shipping = 1 if expedited_shipping.lower() == 'yes' else 0
#     seller_in_us = 1 if seller_in_us.lower() == 'yes' else 0
    
#     # Encode metal type
#     metal_type_encoded = {'Gold': 0, 'Silver': 1, 'Copper': 2, 'Bronze': 3}[metal_type]
    
#     # Encode time
#     time_encoded = {'Less_than_week': 0, 'Less_than_month': 1, 'More_than_year': 2}[time]
    
#     # Encode type of shipping
#     type_of_shipping_encoded = {'Free': 0, 'Calculated': 1, 'FreePickup': 2, 'Auction': 3, 'FixedPrice': 4}[type_of_shipping]
    
#     # Encode type of selling
#     type_of_selling_encoded = {'Auction': 0, 'Fixed Price': 1}[type_of_selling]
    
#     # Encode emperor name
#     emperor_name_encoded = {'Valerian': 0, 'Philip': 1, 'Commodus': 2, 'Domitian': 3, 'Augustus': 4,
#                             'Nero': 5, 'Septimius_Severus': 6, 'Macrinus': 7, 'Otho': 8, 'Nerva': 9,
#                             'Tiberius': 10}[emperor_name]
    
#     return [auto_pay, top_rated_listing, best_offer_enabled, buy_it_now_available,
#             expedited_shipping, seller_in_us, metal_type_encoded, time_encoded,
#             type_of_shipping_encoded, type_of_selling_encoded, emperor_name_encoded]


# # Define routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/museums')
# def museums():
#       return render_template('museums.html') 

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get form inputs
#     auto_pay = request.form['autoPay']
#     top_rated_listing = request.form['topRatedListing']
#     best_offer_enabled = request.form['bestOfferEnabled']
#     buy_it_now_available = request.form['buyItNowAvailable']
#     expedited_shipping = request.form['expeditedShipping']
#     seller_in_us = request.form['sellerInUS']
#     metal_type = request.form['metalType']
#     time = request.form['time']
#     type_of_shipping = request.form['typeOfShipping']
#     type_of_selling = request.form['typeOfSelling']
#     emperor_name = request.form['emperorName']
    
#     # Preprocess form inputs
#     features = preprocess_inputs(auto_pay, top_rated_listing, best_offer_enabled, buy_it_now_available,
#                                  expedited_shipping, seller_in_us, metal_type, time, type_of_shipping,
#                                  type_of_selling, emperor_name)
   

   
    

#     # Make prediction using the model
#     df = pd.DataFrame([features], columns=['autoPay', 'topRatedListing', 'bestOfferEnabled', 'buyItNowAvailable',
#                                        'expeditedShipping', 'sellerInUS', 'metalType', 'time', 'typeOfShipping',
#                                        'typeOfSelling', 'emperorName'])

#     prediction = rf_model.predict(df)
    
#     return render_template('result.html', prediction=prediction[0])

# if __name__ == '__main__':
#     app.run(debug=True)




