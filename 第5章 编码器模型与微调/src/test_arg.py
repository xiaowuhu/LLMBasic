
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", default='../model/ch20/', type=str, 
        help="The output directory where the model checkpoints and predictions will be written.",
    )
parser.add_argument("--bool", default="True", type=bool, help = "Whether to pirnt sth.")
# 这里的bool是一个可选参数，返回给args的是 args.bool
args = parser.parse_args()

if args.bool:
    print('bool = 1')

print(args.output_dir)
