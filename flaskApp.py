from flask import Flask,render_template,request
import modelEvaluation as me
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('Index.html',predictedLabel="")

@app.route('/',methods=['POST'])
def main():
    if 'video' not in request.files:
        return 'no video found'
    video=request.files['video']
    if video.filename=="":
        return 'no video file selected'
    path='static/videos/'+video.filename
    video.save(path)
    predictedLabel=me.predict(path)
    return render_template('Index.html',predictedLabel=predictedLabel)

if __name__=="__main__":
    app.run(debug=True,host="localhost")