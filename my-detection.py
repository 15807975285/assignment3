import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("~/Desktop/raccoon-185.jpg")
display = jetson.utils.videoOutput("111.jpg") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	print(detections[0]) 
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
