import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time gaze estimation demo')
    parser.add_argument('camera_idx', type=int, help='OS index of camera to use.')
    parser.add_argument('fovx', type=float, nargs='?', help='Horizontal FOV in degrees of camera. Does not matter if calib file is given.', default=45.0)
    parser.add_argument('-c', '--calibration', action='store', nargs='?', help="Use camera calibration.", const="data/pseye_calibration.yml")
    parser.add_argument('--cpu', action='store_true', help='Only use the CPU (disables GPU acceleration).')
    parser.add_argument('-m', '--model', type=str, help='Specify a model path (pth). Defaults to default path specified in source.', default=None)
    args = parser.parse_args()
    camera_idx = args.camera_idx
    fovx = args.fovx
    cpu = args.cpu

    import cv2
    from gazeirislandmarks import GazeDetector, FaceNotDetectedError
    from gazeirislandmarks.utilities.capture import FaceCamera
    import matplotlib.pyplot as plt
    from pynput import keyboard
    from PIL import Image

    print("-*-*-*-Controls-*-*-*-")
    print("ESCAPE: Quit")
    print("ENTER: Continue if paused/in save mode")
    print("SPACE: Enter save mode (pause). Pressing SPACE in save mode will save the current image under the 'capture.jpg' name.")

    calibration_file = args.calibration

    cap = FaceCamera(camera_idx, fovx=fovx)
    if calibration_file is not None:
        cap.load_calibration_values(calibration_file)
        print(f"Loaded camera calibration file: {calibration_file}")
    
    detector = GazeDetector(model_path=args.model, device="cuda" if not cpu else "cpu")
    if args.model:
        print(f"Loaded model: {args.model}")

    plt.ion()
    fig = plt.figure()
    _, im = cap.read()
    figim = plt.imshow(im)

    im_mod = None

    with keyboard.Events() as events:
        while True:
            event = events.get(0.0001)
            if event is not None:
                if event.key == keyboard.Key.esc:
                    break
                elif event.key == keyboard.Key.space:
                    event = events.get() # blocking
                    if event.key == keyboard.Key.esc:
                        break
                    elif event.key == keyboard.Key.space:
                        # save
                        Image.fromarray(im_mod).save("capture.jpg")
                    # continue if any other key

            # process
            _, im = cap.read()
            try:
                data = detector(im, cap.M, cap.D, visualize_directions=True, visualize_landmarks=True)
                im_mod = data["visualization"]
                face_pos = data["face_position_in_m"]
                print(f"Face position in meters: {face_pos}")
            except FaceNotDetectedError:
                im_mod = im
            
            figim.set_data(im_mod)
            fig.canvas.flush_events()
            plt.draw()
            