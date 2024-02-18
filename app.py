from re import DEBUG
from flask import Flask,render_template,session,redirect,url_for,request
import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import operator
import numpy as np 
app=Flask(__name__)
req_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']
@app.route('/')
def index():
    return render_template("index.html")
vehicles_count={}
vehicles_count_IMAGE={}
# @app.route("/upload1",methods=["POST","GET"])
# def upload1():
#     try:
#         if request.method=="POST":
#             myfile = request.files['file']
#             print('#############################')
#             global fn1
#             fn1 = myfile.filename
#             mypath = os.path.join('C:/Users/Varma06/OneDrive/Desktop/TK130266/CODE/static/img/team/', fn1)
#             global image1
#             image1=mypath
#             print(image1)
#             print('#############################')
#             myfile.save(mypath)
#             im = cv2.imread(mypath)
#             bbox, label, conf = cv.detect_common_objects(im, model='yolov3.weights')
#             output_image = draw_bbox(im, bbox, label, conf)
#             plt.imshow(output_image)
#             plt.savefig(mypath)
#             count = [label.count(c) for c in req_classes]
#             global count1
#             count1 = sum(count)
#             print('The first uploaded image count is:', count1)
#             vehicles_count["1st Image"]=count1
#             vehicles_count_IMAGE[fn1]=count1
#             print(vehicles_count_IMAGE)
#             return redirect(url_for('upload2'))
#     except:
#         return render_template("uploadimages.html",msg="fail")
#     return render_template("uploadimages.html")

@app.route("/upload1", methods=["POST", "GET"])
def upload1():
    try:
        if request.method == "POST":
            myfile = request.files['file']
            print('#############################')
            global fn1
            fn1 = myfile.filename
            mypath = os.path.join('C:/Users/Varma06/OneDrive/Desktop/TK130266/CODE/static/img/team/', fn1)
            global image1
            image1 = mypath
            print(image1)
            print('#############################')
            myfile.save(mypath)

            # Read the uploaded image
            im = cv2.imread(mypath)

            # Perform object detection using YOLOv3
            net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Prepare the image for detection
            blob = cv2.dnn.blobFromImage(im, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)

            # Get the output layer names
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # Forward pass for object detection
            detections = net.forward(output_layers)

            # Process detections and count objects (adjust this logic based on your class labels)
            count = 0
            for detection in detections:
                for obj in detection:
                    scores = obj[5:]
                    class_id = np.argmax(scores)
                    if class_id == 0:  # Assuming class 0 represents objects you want to count
                        count += 1

            print('The first uploaded image count is:', count)

            # Store the count in a dictionary or wherever required
            vehicles_count["1st Image"] = count
            vehicles_count_IMAGE[fn1] = count
            print(vehicles_count_IMAGE)

            return redirect(url_for('upload2'))
    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template("uploadimages.html", msg="fail")

    return render_template("uploadimages.html")


@app.route('/upload2',methods=["POST","GET"])
def upload2():
    try:
        if request.method=="POST":
            myfile = request.files['file']
            global fn2
            fn2 = myfile.filename
            mypath = os.path.join('static/img/team/', fn2)
            global image2
            image2=mypath
            myfile.save(mypath)
            im = cv2.imread(mypath)
            bbox, label, conf = cv.detect_common_objects(im, model='yolov3.weights')
            output_image4 = draw_bbox(im, bbox, label, conf)
            plt.imshow(output_image4)
            plt.savefig(mypath)
            count = [label.count(c) for c in req_classes]
            global count2
            count2=sum(count)
            print('The second uploaded image count is:', count2)

            vehicles_count["2nd Image"]=count2
            vehicles_count_IMAGE[fn2]=count2
            return redirect(url_for("upload3"))
    except:
        return render_template("upload2.html",msg="fail")
    return render_template("upload2.html")

@app.route('/upload3',methods=["POST","GET"])
def upload3():
    try:
        if request.method=="POST":
            myfile = request.files['file']
            global fn3
            fn3 = myfile.filename
            mypath = os.path.join('static/img/team/', fn3)
            global image3
            image3 = mypath
            myfile.save(mypath)
            im = cv2.imread(mypath)
            bbox, label, conf = cv.detect_common_objects(im, model='yolov3.weights')
            output_image4 = draw_bbox(im, bbox, label, conf)
            plt.imshow(output_image4)
            plt.savefig(mypath)
            count = [label.count(c) for c in req_classes]
            global count3
            count3=sum(count)
            print('The third uploaded image count is:', count3)

            vehicles_count["3rd Image"]=count3
            vehicles_count_IMAGE[fn3]=count3
            return redirect(url_for('upload4'))
    except:
        return render_template("upload3.html",msg="fail")

    return render_template("upload3.html")

@app.route('/upload4',methods=["POST","GET"])
def upload4():
    try:
        if request.method=="POST":
            myfile = request.files['file']
            global fn4
            fn4 = myfile.filename
            mypath = os.path.join('static/img/team/', fn4)
            global image4
            image4 = mypath
            myfile.save(mypath)
            im = cv2.imread(mypath)
            bbox, label, conf = cv.detect_common_objects(im, model='yolov3.weights')
            output_image4 = draw_bbox(im, bbox, label, conf)
            plt.imshow(output_image4)
            plt.savefig(mypath)
            count = [label.count(c) for c in req_classes]
            global count4
            count4=sum(count)
            print('The fourth uploaded image count is:', count4)

            vehicles_count["4th Image"]=count4
            vehicles_count_IMAGE[fn4]=count4
            global  max_key
            max_key = max(vehicles_count_IMAGE.items(), key=operator.itemgetter(1))[0]
            global images_name
            images_name=[]
            for i in range(4):
                Keymax = max(vehicles_count, key=vehicles_count.get)
                del vehicles_count[Keymax]
                images_name.append(Keymax)

            return render_template("success.html",mh="hi")
    except:
        return render_template("upload4.html",msg="fail")

    return render_template("upload4.html")


@app.route('/Viewimage1')
def Viewimage1():

    return render_template("Viewimage1.html",image1=fn1,l1=count1)

@app.route('/Viewimage2')
def Viewimage2():
    print(fn2,'fn2')
    return render_template("Viewimage2.html",image2=fn2,l2=count2)

@app.route('/Viewimage3')
def Viewimage3():
    print(fn3,'fn3')
    return render_template("Viewimage3.html",image3=fn3,l3=count3)

@app.route('/Viewimage4')
def Viewimage4():
    return render_template("Viewimage4.html",image4=fn4,img=images_name,l4=count4)


@app.route('/Viewimage11')
def Viewimage11():
    return render_template("Viewimagey1.html",image1=fn1,l1=count1)
@app.route('/Viewimage22')
def Viewimage22():
    print(fn2,'fn2')
    return render_template("Viewimagey2.html",image2=fn2,l2=count2)

@app.route('/Viewimage32')
def Viewimage32():
    print(fn3,'fn3')
    return render_template("Viewimagey3.html",image3=fn3,l3=count3)

@app.route("/Viewprediction")
def Viewprediction():
    c=images_name[0]
    return render_template("Viewprediction.html",img=images_name,c=c,d=max_key)

@app.route('/xjunction')
def xjunction():
    global list1
    a,b,c,d=count1,count2,count3,count4
    list1=[a,b,c,d]
    first=[a,c]
    second=[b,d]
    list2 = sorted(list1)

    if a==b==c==d:
        import random
        index = random.randint(0,3)
        return render_template("xjunction.html", val= index)
        
    if list2[-1] - list2[-2] > 10:
        index = list1.index(list2[-1])
        return render_template("xjunction.html", val = index)
    else:
        index1 = list1.index(list2[-1])
        index2 = list1.index(list2[-2])
        return render_template("xjunction.html", val1 = index1, val2 = index2)
    


    
@app.route('/yjunction')
def yjunction():
    a,b,c=count1,count2,count3
    li=[a,b,c]
    if max(li)==a:
        return render_template('yjunction.html', a='a')
    if max(li)==b:
        return render_template('yjunction.html', b='b')
    if max(li)==c:
        return render_template('yjunction.html', c='c')

    return render_template('yjunction.html')

@app.route("/uploady1",methods=["POST","GET"])
def uploady1():
    try:
        if request.method=="POST":
            print('0000')
            myfile = request.files['file']
            global fn1
            fn1 = myfile.filename
            mypath = os.path.join('static/img/team/', fn1)
            global image1
            image1=mypath
            myfile.save(mypath)
            im = cv2.imread(mypath)
            bbox, label, conf = cv.detect_common_objects(im, model='yolov3.weights')
            output_image = draw_bbox(im, bbox, label, conf)
            plt.imshow(output_image)
            plt.savefig(mypath)
            count = [label.count(c) for c in req_classes]

            global count1
            count1 = sum(count)
            print('The first uploaded y junction image count is:', count1)

            vehicles_count["1st Image"]=count1
            vehicles_count_IMAGE[fn1]=count1
            print(vehicles_count_IMAGE)
            return redirect(url_for('uploady2'))
    except:
        return render_template("uploadimagesy1.html",msg="fail")
    return render_template("uploadimagesy1.html")

@app.route('/uploady2',methods=["POST","GET"])
def uploady2():
    try:
        if request.method=="POST":
            myfile = request.files['file']
            global fn2
            fn2 = myfile.filename
            mypath = os.path.join('static/img/team/', fn2)
            global image2
            image2=mypath
            myfile.save(mypath)
            im = cv2.imread(mypath)
            bbox, label, conf = cv.detect_common_objects(im, model='yolov3.weights')
            output_image4 = draw_bbox(im, bbox, label, conf)
            plt.imshow(output_image4)
            plt.savefig(mypath)
            count = [label.count(c) for c in req_classes]

            global count2
            count2=sum(count)
            print('The second uploaded y junction image count is:', count2)

            vehicles_count["2nd Image"]=count2
            vehicles_count_IMAGE[fn2]=count2
            return redirect(url_for("uploady3"))
    except:
        return render_template("uploady22.html",msg="fail")
    return render_template("uploady22.html")

#
@app.route('/uploady3',methods=["POST","GET"])
def uploady3():
    try:
        if request.method=="POST":
            myfile = request.files['file']
            global fn3
            fn3 = myfile.filename
            mypath = os.path.join('static/img/team/', fn3)
            global image3
            image3 = mypath
            myfile.save(mypath)
            im = cv2.imread(mypath)
            bbox, label, conf = cv.detect_common_objects(im, model='yolov3.weights')
            output_image4 = draw_bbox(im, bbox, label, conf)
            plt.imshow(output_image4)
            plt.savefig(mypath)
            count = [label.count(c) for c in req_classes]
            global count3
            count3=sum(count)
            print('The third uploaded y junction image count is:', count3)

            vehicles_count["3rd Image"]=count3
            vehicles_count_IMAGE[fn3]=count3
            global max_key
            max_key = max(vehicles_count_IMAGE.items(), key=operator.itemgetter(1))[0]
            global images_name
            images_name = []
            for i in range(3):
                Keymax = max(vehicles_count, key=vehicles_count.get)
                del vehicles_count[Keymax]
                images_name.append(Keymax)

            return render_template("success1.html", mh="hi")
    except:
        return render_template("upload3y.html",msg="fail")

    return render_template("upload3y.html")



if (__name__)==('__main__'):
    app.run(debug=True)

