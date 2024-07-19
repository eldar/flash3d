import json
import pickle

def transform_json_to_txt(json_data, seq_data, src_idx=0):
    result = []
    total_excluded = 0
    total_processed = 0
    for key, value in json_data.items():
        try:
            seq_data[key]
        except:
            print("Didn't find {} in data".format(key))
            total_excluded += 1
            continue

        if value is not None:
            value_element=value
            total_processed += 1
            if src_idx == -1:
                print(value_element)
                for tgt in value_element['target']:
                    if abs(int(tgt) - int(value_element['context'][0])) <= \
                        abs(int(tgt) - int(value_element['context'][1])):
                        src_idx_adjusted = 0
                    else:
                        src_idx_adjusted = 1
                    context = value_element['context'][src_idx_adjusted]
                    target = ' '.join(map(str, [tgt, tgt, tgt]))
                    result.append(f"{key} {context} {target}")
            else:     
                context = value_element['context'][src_idx]
                target = ' '.join(map(str, value_element['target']))
                result.append(f"{key} {context} {target}")
    
    print("Excluded {} sequences. Processed {}".format(total_excluded, total_processed))
    return '\n'.join(result)

def main():

    test_pickle_path = "/scratch/shared/beegfs/shared-datasets/Realestate10k/test.all.pickle"
    with open(test_pickle_path, "rb") as f:
        seq_data = pickle.load(f)

    # Read JSON data from file
    with open('/users/stan/unsup-gauss/splits/re10k_pixelsplat/eval_split_2_frames.json', 'r') as file:
        json_data = json.load(file)

    # Transform JSON to text
    output_text = transform_json_to_txt(json_data, seq_data, src_idx=-1)

    # Write the output text to a file
    with open('test_closer_as_src.txt', 'w+') as file:
        file.write(output_text)

    # Transform JSON to text with the other source view
    # output_text = transform_json_to_txt(json_data, seq_data, src_idx=1)

    # # Write the output text to a file
    # with open('test_second_as_src.txt', 'w+') as file:
    #     file.write(output_text)
    

if __name__ == "__main__":
    main()