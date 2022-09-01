#!/bin/bash
# Author: Tomas Goldsack


fileid="1lvsqvsFi3W-pE1SqNZI0s8NR9rC1tsja"
filename="pubmed-data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip ${filename}

fileid="1bUczUFivhTLOBj6Qb2e7Q8HICBXLqlyi"
filename="eLife_split.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip ${filename}

fileid="1GJbH3GP3Kc14hvNwcGHyuhLd_MTG9UqQ"
filename="PLOS_split.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip ${filename}

fileid="1Aa7K9KX79ZAQn3LHP2nhw5dkn1oW1ACj"
filename="UMLS_files.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip ${filename}
