# PicABike
An app to find bikes on Craigslist by uploading an image.

Code developed as a fellow at Insight Data Science.

## Dependencies
- Python 3.6 (numpy, pandas, scipy, Keras, Tensorflow, scikit-learn, flask, psycopg2, beautifulsoup, sqlalchemy)
- PostgreSQL


## Algorithm Development and Validation
- `generate_model_data.py` - generate training and test data for model training
- `label_training_postings.py` - label training data based on their folder
- `model.best.hdf5` - model weights for top model
- `predict_bike_types.py` - predict bike types for pulled Craigslist postings
- `pull_bikes_craigslist.py` - scrape Craigslist postings
- `pull_bikes_ebay.py` - pull Ebay postings using Ebay API
- `save_postings.py` - save scraped and pulled postings into a PostgreSQL database
- `train_model.py` - extract imagefeatures from CNN and train last layer + classifier
- `update_bikes_auto.py` - automatically scrape and label new Craigslist postings every three hours
- `validate_against_text.py` - compare search results for different bike types using a text search and new PicABike search

## Flask Application
- `run_flask.py` - run flask app
- `flasklib` - backbones for app
