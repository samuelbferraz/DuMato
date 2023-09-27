echo "Table 6.3"
echo "DFS"

repeat=3
EXEC_DIR=exec
DATASETS_DIR=/srv/samuel.ferraz/datasets
RESULTS_DIR=results

##########
# motifs #
##########

app="motifs"
dataset="citeseer"
k_values=(3 4 5 6 7)
executable="motifs_DM_DFS"
folder="table_6_3"
mkdir $RESULTS_DIR/$folder

for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        echo "./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
	exit
    done
done

app="motifs"
dataset="ca_astroph"
k_values=(3 4)
executable=$EXEC_DIR/"motifs_DM_DFS"
folder="table_6_3"


for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
    done
done

app="motifs"
dataset="mico"
k_values=(3 4)
executable=$EXEC_DIR/"motifs_DM_DFS"
folder="table_6_3"


for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
    done
done

app="motifs"
dataset="dblp"
k_values=(3 4 5)
executable=$EXEC_DIR/"motifs_DM_DFS"
folder="table_6_3"


for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
    done
done

##########
# clique #
##########

app="clique"
dataset="citeseer"
k_values=(3 4 5 6)
executable=$EXEC_DIR/"clique_DM_DFS"
folder="table_6_3"


for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
    done
done

app="clique"
dataset="ca_astroph"
k_values=(3 4 5 6 7 8)
executable=$EXEC_DIR/"clique_DM_DFS"
folder="table_6_3"


for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
    done
done

app="clique"
dataset="mico"
k_values=(3 4 5)
executable=$EXEC_DIR/"clique_DM_DFS"
folder="table_6_3"


for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
    done
done

app="clique"
dataset="dblp"
k_values=(3 4 5 6 7)
executable=$EXEC_DIR/"clique_DM_DFS"
folder="table_6_3"


for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
    done
done

app="clique"
dataset="livejournal"
k_values=(3 4)
executable=$EXEC_DIR/"clique_DM_DFS"
folder="table_6_3"


for k in "${k_values[@]}"
do
    for i in $(seq $repeat); do
        echo "($i): $app,$dataset,$k,$EXEC_DIR/$executable"
        ./$EXEC_DIR/$executable $DATASETS_DIR/$dataset.edgelist $k >> $RESULTS_DIR/$folder/$app.$dataset.$executable.$k.$i.out
    done
done
