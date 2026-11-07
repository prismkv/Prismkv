"""Attach a matching Full trajectory as a scoring target, never as model input."""
import argparse,hashlib,json
from pathlib import Path
from prismkv.evaluation import sha_file


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--full-predictions',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    rows=[json.loads(x) for x in Path(args.inputs).read_text().splitlines() if x.strip()]
    reference=[json.loads(x) for x in Path(args.full_predictions).read_text().splitlines() if x.strip()]
    by_id={r['id']:r for r in reference}
    if len(by_id)!=len(reference) or len({r['id'] for r in rows})!=len(rows) or set(by_id)!={r['id'] for r in rows}:
        raise ValueError('reference/input IDs must match exactly')
    result=[];source_SHA=sha_file(args.full_predictions)
    for row in rows:
        ref=by_id[row['id']]
        digest=hashlib.sha256(json.dumps(row['input_ids'],separators=(',',':')).encode()).hexdigest()
        if ref['mode']!='full' or ref['input_ids_SHA']!=digest or not ref['generated_token_ids']:
            raise ValueError('require an original-backbone Full trajectory on identical input IDs')
        if 'reference_token_ids' in row:raise ValueError('input already contains a reference')
        result.append(dict(row,reference_token_ids=ref['generated_token_ids'],
                           reference_kind='full_generated',reference_source_SHA=source_SHA))
    with Path(args.output).open('x',encoding='utf8') as output:
        for row in result:output.write(json.dumps(row,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
