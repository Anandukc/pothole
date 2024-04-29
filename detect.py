import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
video_path = "/home/anandu/waste_detect/wastevideo1.mp4"
cap = cv2.VideoCapture(video_path)

# Define the window name
window_name = "YOLOv8 Inference"

# Set the size of the window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)  # Set the desired width and height

# Get the video frame properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
output_video_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Check if objects are detected
        if len(results[0].boxes.xyxy.cpu()) > 0:
            print("Hello")
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the annotated frame to the output video file
        out.write(annotated_frame)
        
        # Display the annotated frame
        cv2.imshow(window_name, annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
