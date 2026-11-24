import argparse
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

"""
DistilBERT PII inference script.
Usage:
    python scripts/predict.py --model models/distilbert-small \
        "My name is John Smith and my email is john@example.com."

Example output:
    'John Smith'         type=NAME_STUDENT
    'john@example.com.'  type=EMAIL

    Highlighted text:
    My name is [[NAME_STUDENT: John Smith]] and my email is
    [[EMAIL: john@example.com.]]
    
1. 입력을 받는다.
2. 단어별로 스플릿해서 위치를 기록한다.
3. 토큰화한다.
4. 모델을 이용해서 예측 결과를 받는다.
5. 단어 단위 예측값 통합
6. merge
7. 최종결과 출력
int main() {
    string text = getInput();

    auto [words, offsets] = splitWords(text);

    auto tokens = tokenizer.encode(words);

    auto logits = model.forward(tokens);

    auto labels = aggregatePredictions(logits);

    auto entities = mergeBIO(words, offsets, labels);

    printResults(text, entities);
} 이런 구조다


Python	        C/C++

list	        std::vector
dict	        std::unordered_map 또는 std::map
tuple	        std::tuple, std::pair
set	             std::unordered_set 또는 std::set
None	        값이 없음을 나타내는 상태, std::nullopt와 유사
for x in items	    C++ range-based for
enumerate(items)	인덱스와 원소를 함께 순회
zip(a, b)	        여러 컨테이너를 병렬 순회
len(items)	            items.size()
items.append(x)	            items.push_back(x)
dict[key]	            map[key]
f"{value}"	            문자열 포매팅
a, b = function()	C++17 structured binding과 유사
"""


def split_words(text): # predict에서 호출하는 함수다
    words = []
    offsets = []

    for i in re.finditer(r"\S+", text): # /S+: 공백문자가 아닌 하나이상의 연속문자. 이걸 찾는다 text안에서
        words.append(i.group()) # words배열에 단어 기록
        offsets.append((i.start(), i.end())) # offfsets에 시작 주소와 마지막 주소를 적는다.(개인정보룰 발견시에 거기가 어딘지 알아야해서)

    return words, offsets 


def merge_bio_predictions(words, offsets, labels): # 찾은 개인정보들만 기록하는거임
    entities = []
    current = None # 현재 merge하고 있는 개인정보

    for word, (start, end), label in zip(words, offsets, labels): # 3개의 배열을 동시에 순회하는거임

        if label == "O":
            if current is not None: # 결과를 저장한다.
                entities.append(current)
                current = None
            continue

        if "-" not in label: # 형식에 맞는지 체크
            continue

        prefix, entity_type = label.split("-", 1) #split prefix in the word

        if prefix == "B": # Begin -> its new pii
            if current is not None:
                entities.append(current)

            current = {"type": entity_type, "start": start, "end": end}

        elif prefix == "I":# inner -> add 
            if current is not None and current["type"] == entity_type:
                current["end"] = end

            else:
                if current is not None:
                    entities.append(current)

                current = {"type": entity_type, "start": start, "end": end}

    if current is not None:
        entities.append(current)

    return entities


def highlight_text(text, entities):
    result = ""
    last_end = 0

    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]

        result += text[last_end:start]

        result += f"[[{entity_type}: " f"{text[start:end]}]]"

        last_end = end

    result += text[last_end:]

    return result


def redact_text(text, entities): # masking func
    result = ""
    last_end = 0

    for entity in entities:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]

        result += text[last_end:start]
        result += f"[{entity_type}]"

        last_end = end

    result += text[last_end:]

    return result


def predict(model_path, text): # 1번쨰로 실행되는 함수다.

    device = torch.device( # cpu혹은 gpu를 선택한다.
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Selected Device:", device)

    tokenizer = AutoTokenizer.from_pretrained( # tokenizer를 불러오는 역할이다.
        model_path, # read dir
        use_fast=True
    )

    model = AutoModelForTokenClassification.from_pretrained( # trained distilbert모델을 불러오기
        model_path
    )

    model.to(device) # let model move to device
    model.eval() # set evaluation mode

    words, offsets = split_words(text) # split함수 호출해서 나눈다.

    if not words: # if empty -> end
        return [], 0, 0

    encoding = tokenizer( # execute tokenizer
        words, # word -> num
        is_split_into_words=True, # we already spilted before
        truncation=True, # if too long -> cut, but there is overflowing below
        max_length=512, # window size : 1window = 512 token
        stride=128, # make duplicated area. 예를들어 1~512->window1, (512-128)~(512-128+512) -> window2.
        return_overflowing_tokens=True, # if need -> make more window
        padding="max_length", # 빈 토큰으로 윈도우의 빈칸을 채워서 512개의 토큰으로 만들어버린다?? 왜?
        return_tensors="pt" # 결과를 pytorch tensor로 변환해라.
    )

    model_inputs = {
        key: value.to(device) # move tensor to gpu
        for key, value in encoding.items() # encoding값을 불러온다.
        if key in tokenizer.model_input_names # 필요한거만
    }

    with torch.no_grad(): # we don't need to train now; take off for my memory
        outputs = model(**model_inputs) # **는 모델input들을 말함. forward 연산 실행 outputs에 결과를 저장한다.

    all_logits = outputs.logits.detach().cpu() # 각 클래스에 대해 분류점수들을 보고 softmax처럼 그리고 텐서를 분리하여 cpu로 이동시킨다. 결과집계는 cpu에서

    num_windows = all_logits.shape[0] # num of window/ shape(3,512,5) -> 3windows with 512token each and label : 5

    word_logit_sums = {} # 같은 단어를 합치기 위한 구조다. 각단어의 logit의 합을 저장? n번단어의 벡터값의 합
    word_logit_counts = {} #각 단어의 logit를 몇번 합쳤는지 카운팅? 그게 몇개의 윈도우에서 나왔는가

    for window_index in range(num_windows):

        word_ids = encoding.word_ids( # 이 토큰이 어느 단어에 속하는가 ex) ts@as.com -> word = 1 token = 5
            batch_index=window_index
        )

        seen_in_window = set() # prevent duplication

        for token_index, word_id in enumerate(word_ids):

            if word_id is None: # skip none
                continue

            if word_id in seen_in_window: # skip if we already check same word.
                continue

            seen_in_window.add(word_id) # 처리한것은 더이상 못하도록 visit처리하는거임

            logits = all_logits[ # token의 logit을 가져온다? logit가 뭐지? -? x번 윈도우의y번 토큰에 대한 각label점수들을 말하는거 같다.
                window_index,
                token_index
            ]

            if word_id not in word_logit_sums: # 단어별 logit이 처음 나왔는지를 체크하는 것이다.
                word_logit_sums[word_id] = logits.clone() # 처음이면 logit를 복사해서 저장 갯수를 1로
                word_logit_counts[word_id] = 1
            else:
                word_logit_sums[word_id] += logits # 이미 나왔으면 logit을 더하고 갯수를 증가 이후에 이것들의 평균을 구할거임
                word_logit_counts[word_id] += 1

    word_labels = [] # 최종 label저장
    processed_count = 0 # 실제 예측점수

    for word_id in range(len(words)): # 원본 단어를 전부 순회한다.

        if word_id not in word_logit_sums: # 처리 되지 않은 단어(no logit) -> not pii
            word_labels.append("O")
            continue

        average_logits = ( # 평균을 구한다.
            word_logit_sums[word_id]
            / word_logit_counts[word_id]
        )

        label_id = torch.argmax( # 가장 높은 것을 반환한다.
            average_logits
        ).item()

        label = model.config.id2label[label_id] # num -> name ex) 1-> B-name

        word_labels.append(label) #lable를 추가 ex) ["O", "O", "O", "B-NAME_STUDENT", "I-NAME_STUDENT"]
        processed_count += 1#카운팅

    entities = merge_bio_predictions( #ex) John  → B-NAME_STUDENT --->{ "type": "NAME_STUDENT","start": 11, "end": 21} 왜하는거지?
        words,
        offsets,
        word_labels
    )

    print("Windows processed:", num_windows) # 처리한 윈도우 수 출력 디버깅용

    return entities, processed_count, len(words) # 최종 결과를 반환한다.


def main():

    parser = argparse.ArgumentParser( # c++: argc, argv 중 argv를 파서에 담는다
        description="Detect PII using a DistilBERT checkpoint"
    )

    parser.add_argument("--model", required=True, help="Path to checkpoint directory") # 모델옵션 --model models/full_baseline

    parser.add_argument("text", help="Text to scan for PII") # text를 저장하는거같음 따로 추출해서
    parser.add_argument( # mask option
        "--redact",
        action="store_true",
        help="Replace detected PII with entity type labels",
    )

    args = parser.parse_args() # make args

    entities, processed, total = predict(args.model, args.text) # 함수 호출한다.

    print("\nDetected PII:")
    print("-" * 70)

    if not entities:
        print("No PII detected.")

    else:
        for entity in entities:

            span = args.text[entity["start"] : entity["end"]] # 개인정보 위치의 문자를 찾는다.

            print(
                f"{span!r:30} "  #r : repr() make S
                f"type={entity['type']:20} "
                f"offset=({entity['start']}, {entity['end']})"
            )

    print("\nHighlighted text:")
    print("-" * 70)
    print(highlight_text(args.text, entities))

    json_entities = []

    for entity in entities:
        json_entities.append(
            {
                "type": entity["type"],
                "text": args.text[entity["start"] : entity["end"]],
                "start": entity["start"],
                "end": entity["end"],
            }
        )

    print("\nJSON output:")
    print("-" * 70)
    print(json.dumps(json_entities, indent=2, ensure_ascii=False))

    if args.redact:
        print("\nRedacted text:")
        print("-" * 70)
        print(redact_text(args.text, entities))




if __name__ == "__main__":
    main()
