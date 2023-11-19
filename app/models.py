# app/models.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class images(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=True)
    image = db.Column(db.LargeBinary)
    ship_count = db.Column(db.Integer)
    product = db.Column(db.String(200))

    def __init__(self, name, image, ship_count, product):
        self.name = name
        self.image = image
        self.ship_count = ship_count
        self.product = product

class ships(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    number = db.Column(db.String(200), nullable=False)
    classification = db.Column(db.String(200))
    size = db.Column(db.Integer)

    def __init__(self, number, classification, size):
        self.number = number
        self.classification = classification
        self.size = size
