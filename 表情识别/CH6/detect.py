import torch
import utils
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from NewResNet import resnet18

config = utils.read_config()

# 实例化自己的网络并载入训练参数

data_trans = transforms.Compose([transforms.Resize([224, 224]),
                                 transforms.ToTensor()])
label_name = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
device = config['device']
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

num_classes = config['num_classes']
device = config['device']
model = resnet18(num_classes).to(device)
# 网络模型已经加载，请在此处加载权重路径对应的权重文件
# 加载模型超参数
model.load_state_dict(torch.load(config['model_path'], map_location=device))
model.eval()
print("load model successfully")

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                                minNeighbors=5)
    # for (x, y, w, h) in faces_rects:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     face_img = gray[y: y + h, x: x + w]
    #     face_img = np.tile(face_img, 3).reshape((w, h, 3))
    #     face_img = Image.fromarray(face_img)
    #     face_img = data_trans(face_img) 
    #     face_img = face_img.unsqueeze(0)
    #     face_img = face_img.to(device)
    #     np_face_img = face_img.cpu().numpy()
    #     label_pd = model(face_img)
    #     predict_np = np.argmax(label_pd.cpu().detach().numpy(), axis=1)
    #     fer_text = label_name[predict_np[0]]
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     pos = (x, y)
    #     font_size = 1.5
    #     color = (0, 0, 255)
    #     thickness = 2
    #     cv2.putText(frame, fer_text, pos, font, font_size, color, thickness,
    #                 cv2.LINE_AA)

    for (x, y, w, h) in faces_rects:
        x_lu, y_lu = max(0, x-config['extend_length']), max(0, y-config['extend_length'])
        height, width, _ = frame.shape
        x_rd, y_rd = min(width, x+config['extend_length']+w), min(height, y+config['extend_length']+h)
        cv2.rectangle(frame, (x_lu, y_lu), (x_rd, y_rd), (0, 255, 0), 2)
        face_img = frame[y_lu:y_rd, x_lu:x_rd]
        face_img = Image.fromarray(face_img)
        face_img = data_trans(face_img)
        face_img = face_img.unsqueeze(0)
        face_img = face_img.to(device)
        label_pd = model(face_img)
        predict_np = np.argmax(label_pd.cpu().detach().numpy(), axis=1)
        fer_text = label_name[predict_np[0]]
        font = cv2.FONT_HERSHEY_SIMPLEX
        pos = (x, y)
        font_size = 1.5
        color = (0, 0, 255)
        thickness = 2
        cv2.putText(frame, fer_text, pos, font, font_size, color, thickness,
                    cv2.LINE_AA)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 使用 release()方法释放摄像头，并使用 destroyAllWindows()方法关闭所有窗口
cap.release()
cv2.destroyAllWindows()