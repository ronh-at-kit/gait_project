import argparse


from data_preprocessing.preprocess import preprocess_tumgaid


if __name__ == '__main__':
    example_text = '''example:
    
    use --example 1 to only build first person file to give you a preview
    python main_preprocessing.py --dataset tum --data-root tumgaid/TUMGAIDimage -o tumgaid/TUMGAIDimage_flow/ --example 1
    
    '''

    parser = argparse.ArgumentParser('preprocessing for datasets', epilog=example_text)
    parser.add_argument('--dataset', type=str, help='which dataset to preprocess')
    parser.add_argument('--data-root', type=str, help='dataset root directory')
    parser.add_argument('-o', type=str, help='outputfolder', dest='output_folder')
    parser.add_argument('--example', type=int, help='will only do a single person to give a preview', default=0)

    args = parser.parse_args()
    args.example = args.example > 0

    if args.dataset == 'tum':
        preprocess_tumgaid(args.data_root, args.output_folder, args.example)
        pass