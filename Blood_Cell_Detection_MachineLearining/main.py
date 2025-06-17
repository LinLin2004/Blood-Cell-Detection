import argparse
from train_classifier import train_model
from detect_and_eval import run_detection_and_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'detect'], required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./models/svm_model.pkl')
    parser.add_argument('--feature_type', type=str, default='hog')
    parser.add_argument('--model_type', type=str, choices=['svm', 'rf'], default='svm')
    parser.add_argument('--iou_thresh', type=float, default=0.5, help='IOU threshold for evaluation')
    args = parser.parse_args()

    if args.mode == 'train':
        train_model(args.data_dir, args.model_path, feature_type=args.feature_type, model_type=args.model_type)
    elif args.mode == 'detect':
        run_detection_and_evaluation(args.data_dir, args.model_path, feature_type=args.feature_type, iou_thresh=args.iou_thresh)

if __name__ == '__main__':
    main()


