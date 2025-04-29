import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import time
from djitellopy import Tello

# Initialize the Tello drone
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

# Start the video stream
tello.streamon()
print("Video stream started!")

# Wait for the camera to stabilize
time.sleep(2)

# MediaPipe HandLandmarker setup
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,  # Focus on one hand for better accuracy
    min_hand_detection_confidence=0.3,  # Lower threshold to detect more hands
    min_hand_presence_confidence=0.3,  # Lower threshold
    min_tracking_confidence=0.3  # Lower threshold
)
detector = vision.HandLandmarker.create_from_options(options)

# Helper function to draw landmarks
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    for hand_landmarks in hand_landmarks_list:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )
    return annotated_image

# Simpler, more reliable function to detect thumbs up gesture
def detect_thumbs_up(landmarks):
    # Get the landmarks we need
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index finger tip
    middle_tip = landmarks[12]  # Middle finger tip
    ring_tip = landmarks[16]  # Ring finger tip
    pinky_tip = landmarks[20]  # Pinky tip
    wrist = landmarks[0]  # Wrist
    
    # Check if thumb is extended upward
    thumb_extended_up = thumb_tip.y < wrist.y - 0.05
    
    # Check if other fingers are lower than the thumb
    fingers_down = (
        index_tip.y > thumb_tip.y and
        middle_tip.y > thumb_tip.y and
        ring_tip.y > thumb_tip.y and
        pinky_tip.y > thumb_tip.y
    )
    
    # Return debugging info along with detection result
    is_thumbs_up = thumb_extended_up and fingers_down
    debug_info = {
        "thumb_extended_up": thumb_extended_up,
        "fingers_down": fingers_down,
        "thumb_y": thumb_tip.y,
        "wrist_y": wrist.y,
        "diff": wrist.y - thumb_tip.y
    }
    
    return is_thumbs_up, debug_info

# Simpler, more reliable function to detect thumbs down gesture
def detect_thumbs_down(landmarks):
    # Get the landmarks we need
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index finger tip
    middle_tip = landmarks[12]  # Middle finger tip
    ring_tip = landmarks[16]  # Ring finger tip
    pinky_tip = landmarks[20]  # Pinky tip
    wrist = landmarks[0]  # Wrist
    
    # Check if thumb is extended downward
    thumb_extended_down = thumb_tip.y > wrist.y + 0.05
    
    # Check if other fingers are higher than the thumb
    fingers_up = (
        index_tip.y < thumb_tip.y and
        middle_tip.y < thumb_tip.y and
        ring_tip.y < thumb_tip.y and
        pinky_tip.y < thumb_tip.y
    )
    
    # Return debugging info along with detection result
    is_thumbs_down = thumb_extended_down and fingers_up
    debug_info = {
        "thumb_extended_down": thumb_extended_down,
        "fingers_up": fingers_up,
        "thumb_y": thumb_tip.y,
        "wrist_y": wrist.y,
        "diff": thumb_tip.y - wrist.y
    }
    
    return is_thumbs_down, debug_info

# Main loop
try:
    last_keep_alive = time.time()
    drone_in_air = False
    gesture_cooldown = 0  # Cooldown timer
    gesture_counter = {"thumbs_up": 0, "thumbs_down": 0}  # Counter for consecutive detections
    required_consecutive_detections = 5  # Need this many consecutive detections to trigger action
    
    print("Starting detection loop...")
    
    while True:
        # Capture frame
        frame = tello.get_frame_read().frame
        
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Invalid frame captured, retrying...")
            time.sleep(0.1)
            continue
            
        # Preprocess the frame
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)  # Mirror view
        
        # Enhance brightness and contrast for better detection
        alpha = 1.3  # Contrast (1.0-3.0)
        beta = 15    # Brightness (0-100)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        # Create display frame
        display_frame = frame.copy()
        
        # Check if detection_result and hand_landmarks exist and are not empty
        if detection_result and detection_result.hand_landmarks:
            # Draw hand landmarks
            display_frame = cv2.cvtColor(
                draw_landmarks_on_image(rgb_frame, detection_result),
                cv2.COLOR_RGB2BGR
            )
            
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # Check gestures
            thumbs_up_result, thumbs_up_debug = detect_thumbs_up(hand_landmarks)
            thumbs_down_result, thumbs_down_debug = detect_thumbs_down(hand_landmarks)
            
            # Display debugging info
            cv2.putText(display_frame, f"Thumb-Wrist Y-Diff: {thumbs_up_debug['diff']:.3f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Reset counters if neither gesture is detected
            if not thumbs_up_result and not thumbs_down_result:
                gesture_counter["thumbs_up"] = 0
                gesture_counter["thumbs_down"] = 0
                
            # Process thumbs up
            if thumbs_up_result:
                gesture_counter["thumbs_up"] += 1
                gesture_counter["thumbs_down"] = 0  # Reset other counter
                
                # Show detection progress
                cv2.putText(display_frame, f"Thumbs Up: {gesture_counter['thumbs_up']}/{required_consecutive_detections}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Execute command if enough consecutive detections and cooldown expired
                if gesture_counter["thumbs_up"] >= required_consecutive_detections and time.time() > gesture_cooldown:
                    cv2.putText(display_frame, "EXECUTING: THUMBS UP", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if not drone_in_air:
                        print("Thumbs Up confirmed: Taking off!")
                        tello.takeoff()
                        drone_in_air = True
                    else:
                        print("Thumbs Up confirmed: Moving up")
                        tello.move_up(30)
                        
                    gesture_cooldown = time.time() + 3  # 3-second cooldown
                    gesture_counter["thumbs_up"] = 0  # Reset counter
            
            # Process thumbs down
            if thumbs_down_result:
                gesture_counter["thumbs_down"] += 1
                gesture_counter["thumbs_up"] = 0  # Reset other counter
                
                # Show detection progress
                cv2.putText(display_frame, f"Thumbs Down: {gesture_counter['thumbs_down']}/{required_consecutive_detections}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Execute command if enough consecutive detections, cooldown expired, and drone is in air
                if gesture_counter["thumbs_down"] >= required_consecutive_detections and time.time() > gesture_cooldown and drone_in_air:
                    cv2.putText(display_frame, "EXECUTING: THUMBS DOWN", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    print("Thumbs Down confirmed: Moving down")
                    tello.move_down(30)
                    gesture_cooldown = time.time() + 3  # 3-second cooldown
                    gesture_counter["thumbs_down"] = 0  # Reset counter
        else:
            # No hands detected, reset counters
            gesture_counter["thumbs_up"] = 0
            gesture_counter["thumbs_down"] = 0
            
        # Display status information
        battery = tello.get_battery()
        cv2.putText(display_frame, f"Battery: {battery}%", (10, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if drone_in_air:
            status_text = "Drone: FLYING"
            status_color = (0, 0, 255)
        else:
            status_text = "Drone: LANDED"
            status_color = (255, 0, 0)
            
        cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                   
        # Show cooldown status if active
        if time.time() < gesture_cooldown:
            remaining = int(gesture_cooldown - time.time())
            cv2.putText(display_frame, f"Cooldown: {remaining}s", (10, display_frame.shape[0] - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Keep-alive command
        if time.time() - last_keep_alive > 3:
            tello.send_rc_control(0, 0, 0, 0)  # Send "do nothing" command
            last_keep_alive = time.time()
        
        # Display the frame
        cv2.imshow("Tello Camera Feed", display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('t'):  # Takeoff
            if not drone_in_air:
                print("Taking off!")
                tello.takeoff()
                drone_in_air = True
        elif key == ord('l'):  # Land
            if drone_in_air:
                print("Landing!")
                tello.land()
                drone_in_air = False
        elif key == ord('q'):  # Quit
            break

except KeyboardInterrupt:
    print("Interrupted by user")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()  # Print the full exception details
finally:
    # Land the drone and cleanup
    if drone_in_air:
        try:
            tello.land()
            print("Drone has landed!")
        except:
            print("Error while landing")
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()