i=0
IFS=$'\n'
for line in $(cat $1)
do
    if [ $(($i % 2)) != 0 ]
    then
        echo "$line" | rev | cut -d " " -f 1 | rev
        if [ $((($i + 1)% 10)) == 0 ]
        then
			echo ""
			echo ""
			echo ""
        fi
    fi
    i=$(($i + 1))
done
