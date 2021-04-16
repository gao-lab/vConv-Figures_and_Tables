#!/bin/bash

R CMD Rserve --no-save
cd $1
java -jar $3 $2 12 0.005

