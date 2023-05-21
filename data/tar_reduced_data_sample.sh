#!/bin/bash

### If you have extracted all the cube files elsewhere use this to create a tar.gz
### that contains a sample of the data


cubes_tar=100cubes.tar
if [ -f density.0000000001.cube ] ; then
   gzip density.0000000001.cube
fi

tar cvf $cubes_tar density.0000000001.cube.gz

for num in `seq 100 500 50000` 
do
   cube_file=`printf 'density.%010d.cube' $num`
   echo $cube_file
   if [ -f $cube_file ] ; then
      gzip $cube_file
   fi
   tar rvf $cubes_tar $cube_file.gz
done
gzip $cubes_tar
