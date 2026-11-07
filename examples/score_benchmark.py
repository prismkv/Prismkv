"""Run caller-owned official scorers on exactly paired prepared inputs/results."""
import argparse,json
from pathlib import Path
from prismkv.benchmarks import score_records
from prismkv.evaluation import sha_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--benchmark',choices=['longbench','ruler'],required=True)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--official-root',required=True)
    parser.add_argument('--output',required=True,help='new JSON file only')
    args=parser.parse_args()
    rows=[json.loads(x) for x in Path(args.inputs).read_text().splitlines() if x.strip()]
    predictions=[json.loads(x) for x in Path(args.predictions).read_text().splitlines() if x.strip()]
    result=score_records(rows,predictions,official_root=args.official_root,benchmark=args.benchmark)
    result.update(inputs_SHA=sha_file(args.inputs),predictions_SHA=sha_file(args.predictions))
    with Path(args.output).open('x',encoding='utf8') as output:json.dump(result,output,indent=2,allow_nan=False)


if __name__=='__main__':main()
