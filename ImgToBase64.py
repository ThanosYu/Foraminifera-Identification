import base64

with open("C:\\Users\\z003zdbc\\Desktop\\foraminifer\\NCSU-CUB_Foram_Images_G-bulloides\\02-24-17_Trial_1 G. Bulloides OOP 552A 7-9 cm 250-355 micro\\imgray_1.png", 'rb') as f:
    base64_data = base64.b64encode(f.read())
    s = base64_data.decode()
    print('data:image/jpeg;base64,', s)
