#!/bin/bash

### If you have extracted all the cube files, use this to remove a portion of them from 
### the directory

tmpdir=tmp

while true; do

read -p "This will erase data, do you want to proceed? (y/n) " yn

case $yn in 
	[yY] ) echo deleting...;
		break;;
	[nN] ) echo exiting...;
		exit;;
	* ) echo invalid response;;
esac

done
if [ -d $tmpdir ] ; then
	echo "$tmpdir exists, will not proceed" 
	exit
fi
mkdir -p $tmpdir
for num in `seq 100 500 50000` 
do
   cube_file=`printf 'density.%010d.cube' $num`
   echo $cube_file
   if [ -f $cube_file ] ; then
      mv $cube_file $tmpdir
   fi
done

for file in *.cube
do
	echo $file
	rm $file
done

mv $tmpdir/*.cube .
rm -rf $tmpdir
