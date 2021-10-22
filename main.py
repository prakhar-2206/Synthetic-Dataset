# importing neccesary libraries
import cv2
import json
import random
import math
import pandas as pd
import matplotlib.pyplot as plt


# choosing random college name
def college():
    college_df = pd.read_csv("College.csv")
    college_name = random.choice(list(college_df['College_Name']))
    return college_name


# choosing random student name
def name():
    name_df = pd.read_csv("Name.csv")
    name_ = random.choice(list(name_df['name']))
    return name_


# choosing random branch of study
def branch():
    branch_df = pd.read_csv("Branch.csv")
    branch_name = random.choice(list(branch_df['Branch']))
    return branch_name


# choosing random Date of Birth
def date_of_birth():
    day_ = random.choice([i for i in range(1, 31)])
    month_ = random.choice([i for i in range(1, 13)])
    year_ = random.choice([i for i in range(1990, 2021)])
    date_ = str(day_) + "/" + str(month_) + "/" + str(year_)
    return date_


# choosing random address
def address():
    address_df = pd.read_csv("Address.csv")
    address_ = random.choice(list(address_df['Address']))
    if len(address_) <= details["address_size"]:
        return address_
    return address()


# choosing random validity date for the Id-Card
def valitdity():
    valitdity_ = random.choice([i for i in range(2018, 2023)])
    return str(valitdity_)


# Probablity for adding key for name, dob, branch and address
def choice_for_name_dob_branch_address_key():
    out = random.choice([i for i in range(10)])
    return (out <= 9)


# Probablity for adding DoB
def choice_for_dob():
    out = random.choice([i for i in range(10)])
    return (out <= 9)


# Probablity for adding address
def choice_for_address():
    out = random.choice([i for i in range(10)])
    return (out <= 9)


# Probablity for adding validity of Id-Card
def choice_for_validity():
    out = random.choice([i for i in range(10)])
    return (out <= 9)


# selecting random fonts from OpenCV
def fonts():
    out = random.choice([0, 2, 3, 4])
    return out


# selecting random rotation angle
def rotate_angle():
    global angle_value
    angle_value = random.choice([i for i in range(360)])
    return angle_value


# blur the area containing signature
def blur_sign(img, topLeft, bottomRight):
    x, y = topLeft[0], topLeft[1]
    w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

    ROI = img[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(ROI, (51, 51), 0)
    img[y:y+h, x:x+w] = blur


def rotate_image(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg


def add_background(img, img2):
    x = random.choice([i for i in range(50)])
    y = random.choice([i for i in range(50)])
    x_offset, y_offset = x, y
    y1, y2 = y_offset, y_offset + img.shape[0]
    x1, x2 = x_offset, x_offset + img.shape[1]

    alpha_s = img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img2[y1:y2, x1:x2, c] = (alpha_s * img[:, :, c] +
                                 alpha_l * img2[y1:y2, x1:x2, c])

    return img2


def add_photo(img, photo):
    x = details['photo'][0]
    y = details['photo'][1]
    x_offset, y_offset = x, y
    y1, y2 = y_offset, y_offset + photo.shape[0]
    x1, x2 = x_offset, x_offset + photo.shape[1]

    alpha_s = photo[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        img[y1:y2, x1:x2, c] = (alpha_s * photo[:, :, c] +
                                alpha_l * img[y1:y2, x1:x2, c])

    return img


def id_card(image_path, background_path, photo_path):

    # reading and resizing the id_card image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (details["size"][0], details["size"][1]))

    blur_sign(img, details["top_left_sign"], details["bottom_right_sign"])

    # reading and resizing the backgroung image
    img2 = cv2.imread(background_path, -1)
    img2 = cv2.resize(img2, (1500, 1500))

    # reading and resizing the student photo
    photo = cv2.imread(photo_path, -1)
    photo = cv2.resize(photo, (250, 300))

    # adding photo to the id_card
    add_photo(img, photo)

    font1, font2 = fonts(), fonts()

    # adding random text data (address, college name,
    # student name, dob, etc..) to the id card image
    cv2.putText(img, text=college(), org=(details["college"][0],
                details["college"][1]), fontFace=font1,
                fontScale=details["font_size"], color=(0, 0, 0, 255),
                thickness=details['thickness'], lineType=cv2.LINE_AA)

    cv2.putText(img, text=address(), org=(details["col_address"][0],
                details["col_address"][1]), fontFace=font1, fontScale=1,
                color=(0, 0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    if(choice_for_validity()):
        cv2.putText(img, text="Validity : " + valitdity(),
                    org=(details["validity"][0], details["validity"][1]),
                    fontFace=font2, fontScale=1,
                    color=(0, 0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    if(choice_for_name_dob_branch_address_key()):

        cv2.putText(img, text="NAME : " + name(), org=(details["name"][0],
                    details["name"][1]), fontFace=font2, fontScale=1,
                    color=(0, 0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        cv2.putText(img, text="BRANCH : " + branch(),
                    org=(details["branch"][0], details["branch"][1]),
                    fontFace=font2, fontScale=1,
                    color=(0, 0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        if(choice_for_dob()):
            cv2.putText(img, text="DoB : " + date_of_birth(),
                        org=(details["dob"][0], details["dob"][1]),
                        fontFace=font2, fontScale=1,
                        color=(0, 0, 0, 255), thickness=2,
                        lineType=cv2.LINE_AA)

            if(choice_for_address()):
                cv2.putText(img, text="ADDRESS : " + address(),
                            org=(details["stu_address"][0],
                            details["stu_address"][1] + 50), fontFace=font2,
                            fontScale=1, color=(0, 0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)

        elif(choice_for_address()):
            cv2.putText(img, text="ADDRESS : " + address(),
                        org=(details["stu_address"][0],
                        details["stu_address"][1]),
                        fontFace=font2, fontScale=1, color=(0, 0, 0, 255),
                        thickness=2, lineType=cv2.LINE_AA)

    else:
        cv2.putText(img, text=name(),
                    org=(details["name"][0], details["name"][1]),
                    fontFace=font2, fontScale=1,
                    color=(0, 0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        cv2.putText(img, text=branch(),
                    org=(details["branch"][0], details["branch"][1]),
                    fontFace=font2, fontScale=1,
                    color=(0, 0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        if(choice_for_dob()):
            cv2.putText(img, text="DoB : " + date_of_birth(),
                        org=(details["dob"][0], details["dob"][1]),
                        fontFace=font2, fontScale=1,
                        color=(0, 0, 0, 255),
                        thickness=2, lineType=cv2.LINE_AA)

            if(choice_for_address()):
                cv2.putText(img, text=address(),
                            org=(details["stu_address"][0],
                            details["stu_address"][1] + 50), fontFace=font2,
                            fontScale=1, color=(0, 0, 0, 255),
                            thickness=2, lineType=cv2.LINE_AA)

        elif(choice_for_address()):
            cv2.putText(img, text=address(),
                        org=(details["stu_address"][0],
                        details["stu_address"][1]), fontFace=font2,
                        fontScale=1, color=(0, 0, 0, 255),
                        thickness=2, lineType=cv2.LINE_AA)

    # rotating the id card image by random angle
    img = rotate_image(img, rotate_angle())
    print(angle_value)

    # adding background image to the id card
    img2 = add_background(img, img2)
    image_ = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 4, k+1)
    plt.imshow(image_)
    # plt.imshow(img)


details_image = json.load(open("files.json"))


for k in range(8):

    path = "new\image_" + str(k+1) + ".png"
    filename = details_image[path[4:]]
    file = open(filename)
    details = json.load(file)

    # selecting random background for the id cards
    back = random.choice([i for i in range(1, 10)])
    background_path = "images_backgrounds\img_background_" + str(back) + ".png"

    # selecting random student photo for the id cards
    photo_ = random.choice([i for i in range(1, 9)])
    photo_path = "photos\photo_" + str(photo_) + ".png"

    id_card(path, background_path, photo_path)

plt.show()
