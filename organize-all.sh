
for i in $(ls -d *); do 
	cd $i
	cp ../organize.sh .
	./organize.sh
	cd ..
done




