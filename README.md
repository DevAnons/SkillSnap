<h1 align="center">
SKILLSNAP – AI Basketball Training Application
</h1>

Developed an application on the OutSystems low-code platform that integrates artificial intelligence to enhance the efficiency of basketball training analysis, using MediaPipe Pose for motion analysis and YOLOv11 for basketball detection from user-uploaded training videos.



<h3 align="left">
-- Server Folder -- 🚀
</h3>

Contains the file Server.py which is used for training exercises and running the server. The program usage is as follows:

    1. Download the required libraries for Server.py as listed in the file.
    2. Download all 5 training exercise files from the Models folder and save them in the same folder as Server.py.
    3. Download the BasketballDetect.pt file from the Models folder.
    4. Register and log in to Ngrok Web (https://ngrok.com).
    5. Go to Your Authtoken and click Copy Authtoken.
    6. Open CMD or Windows PowerShell and enter the command: setx NGROK_AUTHTOKEN "your_authtoken_here" Example 
       "NGROK_AUTHTOKEN "323506QfPGqLvDKJQTxdwPyv9Yp_5uzF4Ldp6QWeT3iVPRoHr"
    7. Run the command python Server.py in the PyCharm Terminal (Public Ngrok URL will appear, which can be copied for
       use in the Application).


### -- Models Folder -- 👾
---
#### Contains the following files:
    1. BasketballDetect.pt – a trained model used for basketball detection.
    2. yolov11.pt – a pretrained model for detecting human body key points.
    3. Five training exercise files:
       3.1 Dribble_video.py – for analyzing dribbling.
       3.2 Passing_video.py – for analyzing passing.
       3.3 Shooting_video.py – for analyzing shooting.
       3.4 Defense_video.py – for analyzing defense.
       3.5 Rebound_video.py – for analyzing rebounds.

### -- Application Folder -- 📱
---
#### Contains the OutSystems program named SkillSnap used to create the application. Program usage is as follows:
    1. Log in to OutSystems using a personal environment (create an account if you don’t have one).
    2. Replace the public Ngrok URL obtained from running Server.py 
       (example: "https://branchial-bronchoscopically-emersyn.ngrok-free.dev") in the following locations:
       2.1 Go to Logic → REST → GetDrillsAPI and replace the Base URL.
       2.2 Go to Interface, find "DribbleAnalysis", open the dropdown, double-click the file "CheckWebSocDb",
           then click on the JS file "ConnectWebSoc" and replace the ServerUrl.
       2.3 Repeat step 2.2 for "PassAnalysis", "ShootAnalysis", "DefenseAnalysis", and "ReboundAnalysis",
           then click Publish.
       2.4 Go to Development → Personal URL → Distribute, select Generate Android/iOS app, and a QR code will appear to
           scan and download the application.
       2.5 Use a mobile device to scan the QR code and install the application successfully.
       2.6 You can now use the application (the server must be running first and remain running during application use).
