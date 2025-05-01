import os
import cv2
import time
import argparse
import numpy as np
from ultralytics import YOLO
# setting up argmuments
def detect_faces():
    parser = argparse.ArgumentParser(description='Detect faces using trained YOLOv11 model')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--yaml', type=str, default=None, help='Path to dataset YAML file')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='0', help='Device to use (cpu, 0)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--save', action='store_true', help='Save detection results')
    parser.add_argument('--output', type=str, default=None, help='Output directory for saved results')
    args = parser.parse_args()

    # getting project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # using provided model path or find the best model
    if args.model is None:
        model_path = os.path.join(project_root, 'Data', 'face_recognition', 'weights', 'best.onnx')
        if not os.path.exists(model_path):
            model_path = os.path.join(project_root, 'Data', 'face_recognition', 'weights', 'best.pt')
    else:
        model_path = args.model

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print(f"Loading model from {model_path}")

    # loading class names
    if args.yaml is None:
        dataset_yaml = os.path.join(project_root, 'Dataset', 'face_dataset.yaml')
    else:
        dataset_yaml = args.yaml
    # defaulting to empty dictionary if yaml is not found
    if not os.path.exists(dataset_yaml):
        print(f"Warning: Dataset YAML file not found at {dataset_yaml}")
        class_names = {}  
    else:
        try:
            import yaml
            with open(dataset_yaml, 'r') as f:
                dataset_config = yaml.safe_load(f)
            class_names = dataset_config.get('names', {})
            print(f"Loaded {len(class_names)} class names from {dataset_yaml}")
        except Exception as e:
            print(f"Error loading YAML file: {e}")
            class_names = {}

    # loading model with error handling
    try:
        model = YOLO(model_path)
        print("Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # initialising camera with retries
    max_retries = 3
    retry_count = 0
    cap = None
    
    while retry_count < max_retries:
        try:
            print(f"Opening camera {args.camera} (attempt {retry_count + 1}/{max_retries})")
            cap = cv2.VideoCapture(args.camera)
            
            # checking if camera opened successfully
            # bulit in delay in between checks
            if not cap.isOpened():
                print(f"Failed to open camera {args.camera}")
                cap.release()
                retry_count += 1
                time.sleep(1) 
                continue
                
            # trying to read a test frame
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("Camera opened but failed to read frame")
                cap.release()
                retry_count += 1
                time.sleep(1)
                continue
            # cam is working then exit the retry loop
            print("Camera initialized successfully")
            break  
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            if cap is not None:
                cap.release()
            retry_count += 1
            time.sleep(1)
    
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Failed to initialize camera after {max_retries} attempts")

    # creates output directory if saving results
    if args.save:
        if args.output is None:
            output_dir = os.path.join(project_root, 'Data', 'detections')
        else:
            output_dir = args.output
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving detections to {output_dir}")
        except Exception as e:
			  # disabling saving if directory creation failed
            print(f"Warning: Could not create output directory: {e}")
            args.save = False

    frame_count = 0
    fps_start_time = time.time()
    last_frame_time = time.time()
    consecutive_failures = 0
    max_consecutive_failures = 5

    print("Press 'q' to quit")

    while True:
        try:
            # checking for camera timeouts with three sec interval
            current_time = time.time()
            if current_time - last_frame_time > 3:
                print("Camera timeout detected, attempting to recover...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(args.camera)
                if not cap.isOpened():
                    print("Failed to recover camera")
                    break
                last_frame_time = current_time
                
            # reading frame with error handling
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                print(f"Failed to grab frame ({consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    print("Too many consecutive frame failures, exiting")
                    break
                    
                time.sleep(0.1)
                continue
                
            # reseting failure counter when successful
            consecutive_failures = 0
            last_frame_time = current_time

            # running inference with error handling
            try:
                results = model(frame, conf=args.conf, device=args.device)
            except Exception as e:
                print(f"Inference error: {e}")
                continue

            # processing results
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    try:
                        # etracting detection data
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])

                        # getting class name
                        if cls_id in class_names:
                            class_name = class_names[cls_id]
                        else:
                            class_name = f"Unknown ({cls_id})"

                        # drawing bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # adding label with confidence
                        label = f"{class_name}: {conf:.2f}"
                        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1-label_height-5), (x1+label_width, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    except Exception as e:
                        print(f"Error processing detection box: {e}")
                        continue

            # calculating frames per second
            frame_count += 1
            if frame_count >= 30:
                fps = frame_count / (time.time() - fps_start_time)
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frame_count = 0
                fps_start_time = time.time()

            # displaying frames
            try:
                cv2.imshow('Face Recognition', frame)
            except Exception as e:
                print(f"Display error: {e}")

            # saving results if requested
            if args.save:
                try:
                    timestamp = int(time.time())
                    output_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
                    cv2.imwrite(output_path, frame)
                except Exception as e:
                    print(f"Error saving frame: {e}")

            # checking for exit key with error handling
            try:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("User requested exit")
                    break
            except Exception as e:
                print(f"Key handling error: {e}")

        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            time.sleep(0.5)

    # releasing resources
    print("Cleaning up resources...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped")
# main program loop
if __name__ == "__main__":
    try:
        detect_faces()
    except KeyboardInterrupt:
        print("\nDetection interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
