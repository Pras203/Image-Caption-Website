from flask import Flask, render_template, url_for, redirect, flash

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, Email, ValidationError, EqualTo
from flask_bcrypt import Bcrypt

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

import requests

from io import BytesIO

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///E:/New folder/mp2/mp2/database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'
db=SQLAlchemy(app)

bcrypt=Bcrypt(app)

# Load pretrained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
 
class Caption(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    caption_text = db.Column(db.Text)
    timestamp = db.Column(db.TIMESTAMP, server_default=db.func.current_timestamp())
    user = db.relationship('User', backref=db.backref('captions', lazy=True))

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={'placeholder': 'Username'})
    email = StringField(validators=[InputRequired(), Email(), Length(max=255)], render_kw={'placeholder': 'Email'})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={'placeholder': 'Password'})
    confirm_password = PasswordField(validators=[InputRequired(), EqualTo('password', message='Passwords must match')], render_kw={'placeholder': 'Confirm Password'})
    submit = SubmitField('Register')

    def validate_username(self, username):
        existing_user = User.query.filter_by(username=username.data).first()
        if existing_user:
            raise ValidationError('Username already exists. Please choose a different one.')

    def validate_email(self, email):
        existing_email = User.query.filter_by(email=email.data).first()
        if existing_email:
            raise ValidationError('Email already registered. Please use a different email address.')
class LoginForm(FlaskForm):

    email = StringField(validators=[InputRequired(), Email(), Length(max=255)], render_kw={'placeholder': 'Email'})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={'placeholder': 'Password'})
    submit = SubmitField('Login')

@app.route('/')

def home():
    return render_template('nav.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/caption')
def caption():
    return render_template('caption.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():

        print("in")
 
        user = User.query.filter_by(email=form.email.data).first()

        if user and bcrypt.check_password_hash(user.password, form.password.data):
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password. Please try again.', 'danger') 

    return render_template('login.html', form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    print("abcjd")
    if form.validate_on_submit():
        print("ghgnh")
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')  # Flash a success message
        return redirect(url_for('login'))
    
    else:
        print(form.errors)

    return render_template('signup.html', form=form)

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    image_url = request.json['image_url']
    predictions = predict_step_from_url(image_url)
    return jsonify({'predictions': predictions})  # Return predictions as JSON

def predict_step_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        i_image = Image.open(BytesIO(response.content))
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
    else:
        return []

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)



