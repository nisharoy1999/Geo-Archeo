# main.py
# Single entry point for the entire pipeline.
#
# USAGE:
#   python main.py --stage preprocess
#   python main.py --stage train
#   python main.py --stage evaluate
#   python main.py --stage infer --image data/raw/tile.tif

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Archaeological Site Detection"
    )
    parser.add_argument(
        '--stage',
        choices=[
            'synthetic',
            'preprocess',
            'train',
            'evaluate',
            'infer'
        ],
        required=True
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help="Satellite tile path (infer only)"
    )
    parser.add_argument(
        '--n_samples', type=int, default=120,
        help="Synthetic sample count (synthetic only)"
    )
    args = parser.parse_args()

    if args.stage == 'synthetic':
        print("="*50)
        print("Creating synthetic data for testing")
        print("="*50)
        from create_synthetic_data import main as syn_main
        import sys
        sys.argv = ['create_synthetic_data.py',
                    '--n_samples', str(args.n_samples)]
        syn_main()

    elif args.stage == 'preprocess':
        print("="*50)
        print("STAGE 1: Preprocessing real data")
        print("Input:  data/raw/ + data/labels/")
        print("Output: data/processed/")
        print("="*50)
        from pipeline.preprocess import preprocess_all
        preprocess_all()

    elif args.stage == 'train':
        print("="*50)
        print("STAGE 2: Training")
        print("Input:  data/processed/")
        print("Output: checkpoints/best_model.pt")
        print("="*50)
        from pipeline.train import train
        train()

    elif args.stage == 'evaluate':
        print("="*50)
        print("STAGE 3: Evaluation")
        print("Input:  checkpoints/best_model.pt")
        print("Output: checkpoints/prototypes.pt")
        print("        checkpoints/conformal_threshold.npy")
        print("="*50)
        from pipeline.evaluate import evaluate
        evaluate()

    elif args.stage == 'infer':
        if not args.image:
            print("ERROR: --image required for infer")
            print("python main.py --stage infer "
                  "--image data/raw/tile.tif")
            return
        print("="*50)
        print(f"STAGE 4: Inference on {args.image}")
        print(f"Output: results/detections.pt")
        print("="*50)
        from pipeline.infer import infer
        infer(args.image)


if __name__ == "__main__":
    main()
