import image_net
import face_net

import os
import sys
from socket import *
import threading
import struct

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
FLAGS.image_size = 96
FLAGS.image_color = 3


HOST = '192.168.0.10'
PORT = 5001
BUFFSIZE = 1024
FILE_NUM = 0


# clientIP를 이름으로 하여 디렉토리를 만들어 파일저장
def file_dir(addr):
    global FILE_NUM
    FILE_NUM += 1

    # 파일 이름 형식 : input_00001부터 시작하여 숫자 하나씩 증가시켜 저장
    FILE_NAME = "input_" + str(FILE_NUM).rjust(4, '0') + ".jpg"

    # 현재 프로젝트 폴더의 절대경로
    FILE_PATH = os.path.abspath(os.path.dirname(__file__))

    # 클라이언트 아이피를 받아서 프로젝트 폴더안에 호스트 아이피의 폴더 생성 후 파일을 저장
    clientHOST = str(addr).split(',')[0].replace('(', '').replace("'", '')

    # FILE_PATH = FILE_PATH + "\\" + clientHOST + "\\"
    FILE_PATH = FILE_PATH + "/" + clientHOST + "/"
    FILE_NAME = FILE_PATH + FILE_NAME

    # 폴더가 존재 한다면 만든다
    try:
        if not os.path.exists(FILE_PATH):
            os.makedirs(FILE_PATH)
    except Exception as e:
        print(e)

    return FILE_PATH, FILE_NAME


def handler(clientSocket, addr):
    FILE_PATH, FILE_NAME = file_dir(addr)

    # face_net & image_net 구별 하기 위한 msg 받기
    msg = clientSocket.recv(BUFFSIZE)
    msg = msg.decode()
    print('msg : ', msg)

    file_data = clientSocket.recv(4)
    print(struct.calcsize('i'))
    file_data = struct.unpack('i', file_data)[0]
    print('file_data : ',file_data)
    #final_data = file_data[0]
    #print(final_data)
    f = open(FILE_NAME, 'wb')
    # file_data = str(file_data).encode()
    TOTAL_LEN = 0

    while True:
        data = clientSocket.recv(BUFFSIZE)
        if not data:
            break
        f.write(data)
        TOTAL_LEN += len(data)
        if TOTAL_LEN == int(file_data):
            break
    print('file write is done : ', FILE_NAME)
    f.close()

    if msg == 'imageNet_msg':
        # 이미지 추론중 가장 높은 확률을 갖는 data 추출
        list = image_net.run_inference_on_image(FILE_NAME)
        data = list[0]
        data = bytes(data, encoding='utf-8')
        clientSocket.send(data)
        print('send : ', data)
        clientSocket.close()

    if msg == 'faceNet_msg':
        # build graph
        images = tf.placeholder(tf.float32, [None, FLAGS.image_size, FLAGS.image_size, FLAGS.image_color])
        keep_prob = tf.placeholder(tf.float32)  # dropout ratio
        prediction = tf.nn.softmax(face_net.build_model(images, keep_prob))
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, './session_save/training_save')

        imagefile = FILE_NAME

        # import google.auth
        import io
        # from oauth2client.client import GoogleCredentials
        from google.cloud import vision
        from PIL import Image
        from PIL import ImageDraw

        FLAGS.image_size = 96

        # set service account file into OS environment value
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./visionAPI_yong.json"

        visionClient = vision.Client()
        print('[INFO] processing %s' % (imagefile))

        # detect face
        image = visionClient.image(filename=imagefile)
        faces = image.detect_faces()
        if len(faces)==0:
            print("not detected face")
            clientSocket.send(bytes(
                str('누군지 잘 모르겠어').encode('utf-8')
            ))
            return
        else:
            face = faces[0]

        print('number of faces ', len(faces))

        # get face location in the photo
        left = face.fd_bounds.vertices[0].x_coordinate
        top = face.fd_bounds.vertices[0].y_coordinate
        right = face.fd_bounds.vertices[2].x_coordinate
        bottom = face.fd_bounds.vertices[2].y_coordinate
        rect = [left, top, right, bottom]

        fd = io.open(imagefile, 'rb')
        image = Image.open(fd)

        import matplotlib.pyplot as plt

        # display original image
        print("Original image")
        # plt.imshow(image)
        # plt.show()

        #   draw green box for face in the original image
        print("Detect face boundary box ")
        draw = ImageDraw.Draw(image)
        draw.rectangle(rect, fill=None, outline="green")

        # plt.imshow(image)
        # plt.show()

        crop = image.crop(rect)
        im = crop.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        # plt.show()
        imagefile = imagefile.split('/')[-1]
        im.save(FILE_PATH + 'cropped' + imagefile)

        print("Cropped image")
        tfimage = tf.image.decode_jpeg(tf.read_file(FILE_PATH + 'cropped' + imagefile), channels=3)
        tfimage_value = tfimage.eval()
        tfimages = []
        tfimages.append(tfimage_value)
        # plt.imshow(tfimage_value)
        # plt.show()
        fd.close()

        p_val = sess.run(prediction, feed_dict={images: tfimages, keep_prob: 1.0})
        name_labels = ['iu', 'moretz', 'sana', 'seolhyun', 'wonbin']
        i = 0
        for p in p_val[0]:
            p = p * 100
            print('%s %f' % (name_labels[i], float(p)), '%')
            i = i + 1
        argmaxIndex = p_val[0].argmax()
        data = name_labels[argmaxIndex]

        # 얼굴인식 데이터값 추출
        if len(faces) == 0:
            clientSocket.send(bytes(
                str('누군지 잘 모르겠어').encode('utf-8')
            ))
        else:
            # 데이터 클라이언트로 전달
            data = bytes(data, encoding='utf-8')
            clientSocket.send(data)
            print('send : ', data)
            clientSocket.close()



if __name__ == '__main__':
    ADDR = (HOST, PORT)
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    serverSocket.bind(ADDR)
    serverSocket.listen(5)
    while True:
        print('waiting for connection...')
        clientSocket, addr = serverSocket.accept()
        print('connected from ', addr)
        threading._start_new_thread(handler, (clientSocket, addr))

