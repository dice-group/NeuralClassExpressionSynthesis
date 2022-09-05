import argparse
import json, os
from shutil import copyfile

from manchester_syntax import get_Manchester_Syntax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--example_config", type=str, default="../dl-foil/DLFoil2/ntn.xml", \
                        help="Path to an example configuration file")
    args = parser.parse_args()
    
    kbs = ["carcinogenesis", "mutagenesis", "family-benchmark", "semantic_bible", "vicodi"]
    for kb in kbs:
        path = "../dl-foil/DLFoil2/ontos/"+kb+".owl"
        if not os.path.isfile(path):
            copyfile("../datasets/"+kb+"/"+kb+".owl", "../dl-foil/DLFoil2/ontos/"+kb+".owl")
    for kb in kbs:
        with open("../dl-foil/DLFoil2/"+kb+"_config.xml", "w") as file_config:
            with open("../datasets/"+kb+"/Results/concept_learning_results_ocel.json") as file_lp:
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
                
