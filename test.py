from flask import Flask, render_template, jsonify, request

 
app = Flask(__name__)
       

 
@app.route('/')
def index():
    return render_template('progress_bar.html')
  
@app.route("/ajaxprogressbar",methods=["POST","GET"])
def ajaxprogressbar():
    if request.method == 'POST':
        username = request.form['username']
        # useremail = request.form['useremail']
        # print(username)
        msg = 'New record created successfully'    
    return jsonify(msg)
     
if __name__ == "__main__":
    app.run(debug=True)