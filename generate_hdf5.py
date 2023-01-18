import pickle
from pathlib import Path
import argparse
import uuid
import os.path

from gazeirislandmarks import GazeDetector
from gazeirislandmarks.utilities.general import progress_bar
from gazeirislandmarks.datasets import MPIIFaceGazeDataset, MPIIGazeDataset, GazeCaptureDataset, convert_to_personhdf5, GazeCaptureFazeDataset, UTMultiviewDataset, UTMultiviewFromSynthDataset
from gazeirislandmarks.datasets.columbia import ColumbiaDataset

# import cProfile
# from pstats import Stats, SortKey

def generate_annotations(dataset, path):
    detector = GazeDetector()
    annotations = {}
    for s in progress_bar(dataset):
        p = Path(s["image_path"])
        im_id = str(Path(p.parent.parent.name, p.parent.name, p.name))
        try:
            pose_data = detector.get_headpose_data(s["image"], s["M"], s["D"])
            annotations[im_id] = pose_data
        except:
            print("Error for im_id: %s" % im_id)
    pickle.dump(annotations, open(path, "wb"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility script to generate dataset hdf5 files for training.')
    parser.add_argument('dataset_path', type=str, help='Path to base directory of dataset to process')
    parser.add_argument('dataset_type', type=str, help='Should be either mpiigaze, mpiifacegaze, gazecapture, gazecapturefaze or utmultiview')
    parser.add_argument('output', type=str, help='Output file name')
    parser.add_argument('--annotation_file', nargs='?', const='arg_was_not_given',
            help='Specify an annotations pickled file for normalization or generate one if left unspecified (or use built-in with MPIIGaze.')
    parser.add_argument('--nw', type=int, default=0, help='number of workers')
    parser.add_argument('--full_image', action='store_true', help='Export the full image too.')
    parser.add_argument('--undistort', action='store_true', help='Undistort the images before processing.')
    args = parser.parse_args()

    if args.annotation_file is None:
        annotations = False
    elif args.annotation_file == 'arg_was_not_given':
        # should generate
        annotations = True
        annotation_file = None
        # if args.dataset_type == "mpiifacegaze":
        #     dataset = MPIIFaceGazeDataset(args.dataset_path, undistort=args.undistort)
        #     annotation_file = os.path.join(str(uuid.uuid4()), ".pkl")
        #     generate_annotations(dataset, annotation_file)
        #     dataset = MPIIFaceGazeDataset(args.dataset_path, as_dataloader=not args.full_image, undistort=args.undistort)
        if args.dataset_type == "gazecapture":
            # from .generate_normalized_annotations import generate_annotations
            dataset = GazeCaptureDataset(args.dataset_path, undistort=args.undistort)
            annotation_file = os.path.join(str(uuid.uuid4()), ".pkl")
            generate_annotations(dataset, annotation_file)
            dataset = GazeCaptureDataset(args.dataset_path, as_dataloader=not args.full_image, undistort=args.undistort)
    else:
        # specified
        annotations = True
        annotation_file = args.annotation_file
    # with cProfile.Profile() as pr:
    if args.dataset_type == "mpiifacegaze":
        if annotations and (annotation_file is None):
            dataset = MPIIFaceGazeDataset(args.dataset_path, use_rtgene_normalization=True, as_dataloader=not args.full_image, undistort=args.undistort)
        else:
            dataset = MPIIFaceGazeDataset(args.dataset_path, custom_normalization_path=annotation_file if annotations else None, as_dataloader=not args.full_image, undistort=args.undistort)
    elif args.dataset_type == "mpiigaze":
        dataset = MPIIGazeDataset(args.dataset_path, use_rtgene_normalization=annotations, as_dataloader=not args.full_image, undistort=args.undistort)
    elif args.dataset_type == "gazecapture":
        dataset = GazeCaptureDataset(args.dataset_path, custom_normalization_path=annotation_file if annotations else None, as_dataloader=not args.full_image, undistort=args.undistort)
    elif args.dataset_type == "gazecapturefaze":
        dataset = GazeCaptureFazeDataset(args.dataset_path, as_dataloader=not args.full_image)
    elif args.dataset_type == "utmultiview":
        dataset = UTMultiviewDataset(args.dataset_path, use_rtgene_normalization=True, as_dataloader=not args.full_image) # No other options here
    elif args.dataset_type == "utmultiviewfromsynth":
        dataset = UTMultiviewFromSynthDataset(args.dataset_path, eval_as_last_person=False)
    elif args.dataset_type == "columbia":
        dataset = ColumbiaDataset(args.dataset_path, args.annotation_file)
    else:
        raise ValueError("dataset type is unknown")

    # with open('profiling_stats.txt', 'w') as stream:
    #         stats = Stats(pr, stream=stream)
    #         stats.strip_dirs()
    #         stats.sort_stats('time')
    #         stats.dump_stats('profiling.prof_stats')
    #         stats.print_stats()
    # exit()
    convert_to_personhdf5(dataset, args.output, num_workers=args.nw, image_format="JPEG")
    
    