#!/bin/bash
#if test "$#" -lt 3 then
#    echo "usage: .run_bp.sh edges gt edge_potential"
#    exit 1
#fi

#input parameters
#$1 - edge file
#$2 - gt file
#$3 - edge potential file
${SCRIPT_HOME}/preprocess.sh $1 $2

#calculate the number of nodes
N1=$(awk -F " " '{print $1}' $1 | sort | uniq | wc -l)
N2=$(awk -F " " '{print $2}' $1 | sort | uniq | wc -l)
TOTAL=$(echo $N1 $N2 | awk '{print $1 + $2}')

${SCRIPT_HOME}/bp_hp bp_graph_initial_belief.txt bp_graph_edges.txt $3 $TOTAL 2 bp_graph_results.txt 10
