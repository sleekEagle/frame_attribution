import json

def get_orig_logits(PATH):
    d = {}
    n=0
    with open(PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)   
            except:
                pass         
            d[record['filename']] = record
            n+=1
    return d

d = get_orig_logits(r'C:\Users\lahir\Downloads\data.jsonl')


OUT_PATH = r'C:\Users\lahir\Downloads\partition_32_future_0.0001.jsonl'
for k in d:
     with open(OUT_PATH, 'a') as f:
         f.write(json.dumps(d[k]) + '\n')

pass
