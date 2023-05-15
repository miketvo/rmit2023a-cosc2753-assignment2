import argparse


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--file', required=True, help='reference image')
    ap.add_argument('-r', '--recommender', required=True, help='the machine learning model used for recommendation')
    ap.add_argument('-c', '--classifier', required=True, help='the machine learning model used for classification')
    ap.add_argument('-n', '--num', required=False, default='0', help="number of recommendations, default: 10")
    args = vars(ap.parse_args())
    num = int(args['num'])

    # TODO: Implement this when a recommender is available
