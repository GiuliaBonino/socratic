from datasets import load_dataset
import os
import sys
import re

def copy_dataset(dataset_name, output_folder):
    dataset = load_dataset(dataset_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, row in enumerate(dataset["train"]):
        entry = "- Problem: "+row["Problem"]+"\n- Explanation: "+row["Rationale"].replace("explanation : ","").replace(" answer : option ","").replace(" . answer : ","")[:-2]+"\"\n- Solution: "+[i[1:] for i in re.sub(r'[^a-zA-Z0-9,]', '', row["options"]).replace(" ) ","").split(",") if i[0]==row["correct"]][0]

        print(f"processing file {i+1}")
        file_path = os.path.join(output_folder, f"file{i+1}")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(entry)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <dataset_name> <output_folder>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    output_folder = sys.argv[2]

    copy_dataset(dataset_name, output_folder)