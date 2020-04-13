from flask import Flask
from tensorflow import keras
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_login import UserMixin
from flask import Blueprint, render_template, redirect, url_for, request, flash, make_response
from flask_login import login_required, current_user
from flask import Flask, render_template, Response
from flask_login import login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import send_from_directory

import os
os.environ['no_proxy'] = '*'

#Text to speech
from gtts import gTTS

#Image processing
from camera import Camera
import numpy as np
import cv2
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image

#Chatbot
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

#App
app = Flask(__name__, static_folder='static')

app.config['SECRET_KEY'] = "changemelater"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///login.db"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)



#ChatBot
bot = ChatBot("Emotio", logic_adapters=[
        {
            "import_path": "chatterbot.logic.MathematicalEvaluation",
        },
         {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": "chatterbot.comparisons.levenshtein_distance",
            'default_response':'I am sorry, but I do not understand. I am still learning. Ask me something else..',
            'maximum_similarity_threshold':0.6
        }
        ]
        )


trainer = ChatterBotCorpusTrainer(bot)
trainer.train(
    "chatterbot.corpus.english"
)


trainer = ListTrainer(bot)
trainer.train([
    "How are you?",
    "I am good. Just watching movies in the my robot world",
    "That is good to hear",
    "Thank you",
    "You are welcome",
    "Yes",
    "Ask me something",
    "Yes",
    "So tell me something else? I can sing a song or tell a joke"

])

trainer.train([
    "I am feeling sad",
    "Oh no! Do you think watching a cute cat video might help? It always helps me! Try it https://www.youtube.com/watch?v=NhHBElaarrI",
    "I am sad", 
    "Oh no! I am currntly watching cute pandas videos https://www.youtube.com/watch?v=wAEzpwvrveg, do you want to laugh together?",
    "I am feeling happy", 
    "Yaaay! I do not feel emotions, but I like when my human friends are happy!",
    "I am happy", 
    "Yaaay! Being happy is awesome. What makes you happy?",
    "I am angry", 
    "Oh no! Do not be angry, Emotio can tell you a joke. \"What is a robots favorite kind of music? Heavy Metal.\"",
    "Haha",
    "I know I am a funny robot",
    "I feel angry", 
    "Oh no! Do not be angry, Emotio can tell you a joke. \"What is a robots favorite kind of music? Heavy Metal.\"",
    "I feel scared",
    "What makes you scared? Emotio is scared of spiders, they are everywhere in the Web"  
])

trainer.train([
    "What is your favourite movie?",
    "I love movies about robots! I want to be like Samantha from movie Her when I grow up",
    "Who are you?",
    "I am Emotio, I was designed as a project for CCT431 class",
    "Tell me about yourself",
    "I am Emotio, I was designed as a project for CCT431 class",
    "Do you feel emotions?",
    "No, I do not feel emotions, but I am programmed to understand human facial emotions and be a companion",
    "What do you like?",
    "I like being a companion, that's what I was designed for. I also like to play with my robot friends, but shhh, do not tell this to anyone",
    "Are you a robot?",
    "Yes, I am a robot. My name is Emotio",
    "What do you like to do?",
    "I like being a companion, that's what I was designed for. I also like to play with my robot friends, but shhh, do not tell this to anyone",
    "Are robots bad?",
    "Are humans bad?",
    "Tell me a joke",
    "\"What is a robots favorite kind of music? Heavy Metal.\"",
    "You are funny",
    "Thank you! I know I am funny for a robot!",    
    "How old are you?",
    "Robots do not have age, but I am still a baby robot I was designed in late March 2020",  
    "I am bored",
    "Lets watch some funny cat videos",  
    "Sing a song",
    "I am not good at singing, but... Twinkle, twinkle, little star, How I wonder what you are, Up above the world so high, Like a diamond in the sky, Twinkle, twinkle little star, How I wonder what you are...",  
    "I like you",
    "You are an awesome friend too",
    "Awesome",
    "I think you are awesome"
])



#load model
model = model_from_json(open("emotion1.json", "r").read())
#load weights
model.load_weights('emotion1.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

last_emotion = []

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

#to associate cookie with the actual user obkect
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('login'))

    login_user(user, remember=remember)
    return redirect(url_for('profile'))

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signup', methods=['POST'])
def signup_post():
    email = request.form.get('email')
    name = request.form.get('name')
    password = request.form.get('password')

    user = User.query.filter_by(email=email).first()
    if user:
        flash("Email request already exists.")
        return redirect(url_for('signup'))

    new_user = User(email=email, name=name, password=generate_password_hash(password, method="sha256"))
    db.session.add(new_user)
    db.session.commit()

    return redirect(url_for('login'))

# processing images
def gen_frames(): 
 # generate frame by frame from camera
    camera = cv2.VideoCapture(0)  
    while True:
        success, test_img = camera.read()  # read the camera frame
        if not success:
            print("CAMERA FAILED")
            break
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
           
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,255,255),thickness=2)
            roi_gray=gray_img[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])
                
            emotions = ('angry', 'disgusted', 'scared', 'happy', 'sad', 'surprised', 'neutral')
            predicted_emotion = emotions[max_index] 
            last_emotion.append(predicted_emotion)
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            resized_img = cv2.resize(test_img, (1000, 700))
            ret, buffer = cv2.imencode('.jpg', test_img)
            frame = buffer.tobytes()
      
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result """

# processing images
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# stop processing images
@app.route('/video_stop', methods=['POST'])
def video_stop():
    if last_emotion == []:
        return render_template('interaction.html', name=current_user.name, emotion = '')
    else:
        return render_template('interaction.html', name=current_user.name, emotion = last_emotion[-1])

# Plain chat page
@app.route('/interaction')
def interaction():
    return render_template('plaininteraction.html')

#Not used
@app.route('/audio')
def audio():
    def sound():
        text_to_read = "This is just a test using GTTS, a Python package library"
        language = 'en'
        slow_audio_speed = False
        audio_created = gTTS(text=text_to_read, lang=language,
                            slow=slow_audio_speed)
        audio_created.save("static/speech.mp3")

    
#ChatBot
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')    
    return str(bot.get_response(userText)) 

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)