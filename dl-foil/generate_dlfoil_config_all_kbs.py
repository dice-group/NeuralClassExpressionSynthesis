import argparse
import os, sys
import json
from shutil import copyfile

this_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_file_path.split('dl-foil')[0])

from utils.manchester_syntax import get_Manchester_Syntax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_config", type=str, default="./DLFoil2/ntn.xml", \
                        help="Path to an example configuration file")
    args = parser.parse_args()
    
    kbs = ["carcinogenesis", "mutagenesis", "semantic_bible", "vicodi"]
    for kb in kbs:
        path = this_file_path.split('dl-foil')[0]+"dl-foil/DLFoil2/ontos/"+kb+".owl"
        if not os.path.isfile(path):
            copyfile(this_file_path.split('dl-foil')[0]+"datasets/"+kb+"/"+kb+".owl", this_file_path.split('dl-foil')[0]+"dl-foil/DLFoil2/ontos/"+kb+".owl")
    for kb in kbs:
        with open(this_file_path.split('dl-foil')[0]+"dl-foil/DLFoil2/"+kb+"_config.xml", "w") as file_config:
            with open(this_file_path.split('dl-foil')[0]+"datasets/"+kb+"/Results/concept_learning_results_celoe.json") as file_lp:
                lps = json.load(file_lp)["Learned Concept"]
            with open(args.example_config) as file_example:
                example_lines = file_example.readlines()
            i = 0
            for line in example_lines:
                if "<kb>file" in line:
                    file_name = kb+".owl"
                    file_config.write("<kb>file:./ontos/"+file_name+"</kb>")
                    i += 1
                    continue
                file_config.write(line)
                i += 1
                if "\t<targets>" in line:
                    break
            file_config.write("\n")
            for lp in lps:
                if not '⊤' in lp and not '⊥' in lp:
                    file_config.write("\t\t<target>\n")
                    file_config.write("\t\t"+get_Manchester_Syntax(lp)+"\n")
                    file_config.write("\t\t</target>\n")
            file_config.write("\n")
            
            write = False
            while i < len(example_lines):
                if "</targets>" in example_lines[i]:
                    write = True
                if write:
                    file_config.write(example_lines[i])
                i += 1
                
